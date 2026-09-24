"""Цикл обучения и оценка для одной точки эксперимента по ёмкости.

Две вещи здесь критически важны для науки и не являются стилистическим выбором:

* **Критерий — точность, никогда не функция потерь.**  Проверка «loss
  уменьшился» в предыдущем каркасе проходила, пока контрольная модель сидела на
  точности 0.52 (ловушка T1 спецификации курса).  Ничто в этом пакете не
  принимает решений по функции потерь.
* **Точность на обучающей и на отложенной выборке сообщаются вместе** (раздел 5
  спецификации курса, принцип 3).  Потолок ёмкости обрушивает обе; недообучение
  или заучивание открывает разрыв между ними.  Это различие — результат, а не
  диагностика, поэтому запись запуска несёт обе или не несёт ничего.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import IGNORE_INDEX, RunConfig, derive_seed, device_name, resolve_device
from .data import make_dataset, write_count
from .model import build_model

RESULT_SCHEMA = 1
"""Увеличивается всякий раз, когда меняется раскладка записи, чтобы устаревший
кэш отвергался, а не молча переиспользовался."""


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[float, float]:
    """Средняя кросс-энтропия и точность только по позициям ответов."""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for i in range(0, tokens.shape[0], batch_size):
        tk = tokens[i : i + batch_size].to(device)
        tg = targets[i : i + batch_size].to(device)
        logits = model(tk)
        loss_sum += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tg.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        ).item()
        mask = tg != IGNORE_INDEX
        correct += ((logits.argmax(dim=-1) == tg) & mask).sum().item()
        count += int(mask.sum().item())
    if was_training:
        model.train()
    return loss_sum / count, correct / count


def _lr_lambda(cfg: RunConfig):
    warmup = max(1, int(round(cfg.warmup_frac * cfg.steps)))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, cfg.steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg.final_lr_frac + (1.0 - cfg.final_lr_frac) * cosine

    return fn


def _param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def result_path(out_dir, cfg: RunConfig) -> Path:
    return Path(out_dir) / f"{cfg.run_key()}.json"


def load_result(out_dir, cfg: RunConfig) -> Optional[dict]:
    """Вернуть ранее завершённый запуск или ``None``.

    Возобновляемость (чартер B1): прерванная сессия теряет не больше запуска,
    выполнявшегося в этот момент.  Запись с другой схемой или с сохранённой
    идентичностью, которая не совпадает, отвергается, а не переиспользуется.
    """
    if out_dir is None:
        return None
    path = result_path(out_dir, cfg)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("schema") != RESULT_SCHEMA:
        return None
    if record.get("config") != cfg.identity():
        return None
    return record


def run_experiment(
    cfg: RunConfig,
    device: str = "auto",
    out_dir=None,
    eval_every: int = 0,
    progress: bool = False,
    force: bool = False,
) -> dict:
    """Обучить одну конфигурацию и вернуть её запись.

    ``device``, ``out_dir``, ``eval_every``, ``progress`` и ``force`` — аргументы,
    а не поля конфигурации, намеренно: они меняют то, как запуск выполняется, а
    не то, что он вычисляет, поэтому они не должны попадать в ключ запуска.
    """
    if not force:
        cached = load_result(out_dir, cfg)
        if cached is not None:
            return cached

    dev = resolve_device(device)

    # --- данные и модель: строятся на CPU, затем переносятся ----------------
    train_tokens, train_targets = make_dataset(cfg, cfg.n_train, "train")
    eval_tokens, eval_targets = make_dataset(cfg, cfg.n_eval, "eval")
    n_train_eval = min(cfg.n_eval, cfg.n_train)
    train_eval_tokens = train_tokens[:n_train_eval]
    train_eval_targets = train_targets[:n_train_eval]

    model = build_model(cfg, derive_seed(cfg.seed, "init")).to(dev)

    opt = torch.optim.AdamW(
        _param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg))

    batch_gen = torch.Generator()
    batch_gen.manual_seed(derive_seed(cfg.seed, "batches"))

    history = []
    model.train()
    start = time.perf_counter()
    for step in range(cfg.steps):
        idx = torch.randint(0, cfg.n_train, (cfg.batch_size,), generator=batch_gen)
        tk = train_tokens[idx].to(dev)
        tg = train_targets[idx].to(dev)
        logits = model(tk)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tg.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if eval_every and ((step + 1) % eval_every == 0):
            tr_loss, tr_acc = evaluate(model, train_eval_tokens, train_eval_targets, dev)
            ev_loss, ev_acc = evaluate(model, eval_tokens, eval_targets, dev)
            history.append(
                {
                    "step": step + 1,
                    "batch_loss": loss.item(),
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "eval_loss": ev_loss,
                    "eval_acc": ev_acc,
                }
            )
            if progress:
                print(
                    f"  шаг {step + 1:>6}/{cfg.steps}  batch_loss {loss.item():.4f}  "
                    f"train_acc {tr_acc:.4f}  eval_acc {ev_acc:.4f}",
                    flush=True,
                )

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":  # pragma: no cover - в CI нет MPS
        torch.mps.synchronize()
    seconds = time.perf_counter() - start

    train_loss, train_acc = evaluate(model, train_eval_tokens, train_eval_targets, dev)
    eval_loss, eval_acc = evaluate(model, eval_tokens, eval_targets, dev)

    record = {
        "schema": RESULT_SCHEMA,
        "run_key": cfg.run_key(),
        "label": cfg.label(),
        "config": cfg.identity(),
        "seq_len": cfg.seq_len,
        "write_count": write_count(cfg),
        "state_bytes_per_layer": cfg.state_bytes_per_layer,
        "n_parameters": model.n_parameters(),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "history": history,
        # Опорные точки для прочтения числа точности.  «Выдать какое-нибудь
        # значение из контекста» — вырожденное решение, которое описывает
        # ловушка T1; оно набирает ровно 1/(M + D/2), потому что значения внутри
        # примера различны.
        "chance_acc": 1.0 / cfg.n_value_tokens,
        "in_context_acc": 1.0 / cfg.n_pair_blocks,
        # Только для диагностики.  Ни один тест на это не полагается (раздел 6
        # спецификации курса).
        "seconds": seconds,
        "device": device_name(dev),
        "torch_version": torch.__version__,
    }

    if out_dir is not None:
        path = result_path(out_dir, cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
        tmp.replace(path)

    return record

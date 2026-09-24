"""Блокирующая контрольная проверка ``gate`` (инвариант I6 спецификации курса).

    Точность softmax при наименьшем ``M`` превышает 0.95, прежде чем какой-либо
    сетке разрешено запуститься.

Softmax — ветвь без предела ёмкости.  Если она не достигает потолка, то нет
потолка, с которого могли бы упасть линейные ветви, нет точки коллапса, которую
можно найти, и нет константы, которую можно аппроксимировать, — каждое число,
которое произвело бы домашнее задание, было бы измерением оптимизатора, а не
состояния (ловушка T1 спецификации курса).

**У проверки ``gate`` две точки, а не одна.**  I6 называет наименьшее ``M``;
измерение (см. ``docs/superpowers/specs/2026-09-05-homework-mqar-capacity.md``,
раздел 10) показало, что реальный режим отказа контроля — при *наибольшем* ``M``
перебора, где обучающий сигнал самый скудный относительно числа сохранённых пар.
Проверка только при наименьшем ``M`` проходит с 1.0000, тогда как те же
гиперпараметры набирают 0.009 на верху оси, и это оставило бы весь график
ёмкости неинтерпретируемым.  Поэтому проверяются оба конца.

Проверка идёт по точности **на отложенной выборке**.  Проверку по точности на
обучающей выборке прошла бы модель, которая просто всё заучила.  Точность на
обучающей выборке записывается рядом, и большой разрыв в точке проверки — сам
по себе повод остановиться.

Ничто здесь не носит рекомендательный характер: :func:`require_gate` бросает
исключение, каждая точка входа перебора вызывает её, прежде чем что-либо
конструировать, и нет флага, который её пропускает.  ``force`` лишь сбрасывает
кэш и перезапускает её.
"""

from __future__ import annotations

from typing import Dict, Optional

from .config import RunConfig
from .data import pad_for_seq_len
from .train import run_experiment

#: Точность на отложенной выборке, которую контроль должен превысить.  Заявлена
#: в инварианте I6 спецификации курса.
GATE_THRESHOLD = 0.95

#: Длина последовательности, при которой работает каждая точка перебора по
#: ёмкости.  ``M`` меняется; ``n_pad`` поглощает разницу, так что ``N`` постоянна
#: вдоль оси ёмкости и не коллинеарна с ``M`` (ловушка T3, инвариант I7).
SWEEP_SEQ_LEN = 128

#: Запросов на пример.  Фиксировано на весь перебор — если позволить ``Q``
#: следовать за ``M``, на оси ёмкости появится вторая переменная (инвариант I7),
#: и число записей ``W = 2M + D + Q`` будет двигаться сразу по двум причинам.
#:
#: ``Q`` не свободно.  Измерено: контроль выучивает задачу, только пока число
#: блоков пар ключ/значение, стоящих в контексте, удовлетворяет примерно
#: ``M + D/2 <= 4Q``; дальше он никогда не покидает плато «выдать токен-значение
#: равномерно», и не спасают ни в 4 раза больше шагов, ни в 10 раз больший шаг
#: обучения (learning rate).  Поэтому ``Q`` задаёт верх оси ``M``, и диапазон
#: ``M`` перебора равен ``[Q, 4Q]``.
N_QUERIES = 8

#: Концы оси ёмкости.  ``Q <= M``, потому что запросы — это различные настоящие
#: ключи, выбранные без возвращения, так что наименьшее ``M`` равно ``Q``;
#: наибольшее равно ``4Q`` — измеренная выше граница обучаемости.
SMALLEST_M = N_QUERIES
LARGEST_M = 4 * N_QUERIES

#: Наибольшее число дистракторных токенов, которое тест на число записей может
#: использовать при ``M = SMALLEST_M``, из той же границы ``M + D/2 <= 4Q``.
LARGEST_D = 2 * (LARGEST_M - SMALLEST_M)

GATE_POINTS = ("smallest_m", "largest_m")

#: Замороженные гиперпараметры, в одном месте, чтобы ничто не разъехалось между
#: проверкой ``gate``, перебором и текстом задания.  Каждое число измерено;
#: свидетельства — в таблице раздела 9 спецификации.
#:
#: Три из них критически важны, и их нельзя урезать без повторного измерения:
#:
#: ``short_conv_size=4``
#:     Каузальная depthwise-свёртка на q/k/v.  Без неё эта модель набирает 0.24
#:     при M=4 — в точности ловушка T1 — и никакая ширина, шаг обучения или
#:     число шагов этого не исправляют.  В этом и состоял весь дефект T1.
#: ``steps=2000``
#:     1000 шагов проходят проверку ``gate`` при наименьшем M и напрочь
#:     проваливают её при наибольшем (0.009).  Число шагов — самое дешёвое, что
#:     можно урезать, и первое, что ломает верх оси.
#: ``n_heads=1``
#:     Сохраняет теорию части 1 однозначной: ``d`` в ``SNR = d/(W-1)`` — это
#:     размерность головы, а H голов умножили бы измеренную ёмкость на
#:     необъяснённый множитель H.
BASE_CONFIG = RunConfig(
    # задача — M и D задаются для каждой точки перебора; P выводится из SWEEP_SEQ_LEN
    n_pairs=SMALLEST_M,
    n_distractors=0,
    n_pad=0,
    n_queries=N_QUERIES,
    n_key_tokens=256,
    n_value_tokens=256,
    # модель
    mixer="softmax",
    d_model=128,
    n_layers=2,
    n_heads=1,
    d_k=64,
    d_v=64,
    mlp_ratio=4,
    short_conv_size=4,
    use_pos_emb=True,
    chunk_size=32,
    full_attention_layers=(),
    # оптимизация
    steps=2000,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.1,
    warmup_frac=0.1,
    final_lr_frac=0.05,
    grad_clip=1.0,
    n_train=32768,
    n_eval=2048,
    seed=0,
)


def sweep_config(
    n_pairs: int = SMALLEST_M,
    n_distractors: int = 0,
    seed: int = 0,
    **overrides,
) -> RunConfig:
    """Точка перебора при замороженных гиперпараметрах, дополненная до ``N`` перебора.

    Удержание ``N`` фиксированной с помощью паддинга — это то, что не даёт ``M``
    быть коллинеарным с длиной последовательности (ловушка T3) и оставляет одну
    независимую переменную на оси ёмкости (инвариант I7).  Каждый вызывающий
    проходит через эту функцию, чтобы эта дисциплина жила в одном месте, а не в
    каждом раннере.
    """
    seq_len = overrides.pop("seq_len", SWEEP_SEQ_LEN)
    overrides.pop("n_pad", None)  # P выводится из seq_len, никогда не передаётся снаружи
    cfg = BASE_CONFIG.replace(
        n_pairs=n_pairs, n_distractors=n_distractors, n_pad=0, seed=seed, **overrides
    )
    return cfg.replace(n_pad=pad_for_seq_len(cfg, seq_len))


def gate_config(point: str = "smallest_m", seed: int = 0) -> RunConfig:
    """Конфигурация контроля на одном из концов оси ёмкости."""
    if point == "smallest_m":
        return sweep_config(n_pairs=SMALLEST_M, seed=seed)
    if point == "largest_m":
        return sweep_config(n_pairs=LARGEST_M, seed=seed)
    raise ValueError(f"неизвестная точка проверки gate {point!r}; ожидается одна из {GATE_POINTS}")


#: Точка, которую называет инвариант I6, вынесена в константу модуля для удобства.
GATE_CONFIG = gate_config("smallest_m")


class GateFailure(RuntimeError):
    """Контроль не достиг потолка.  Ничто дальше по конвейеру запускаться не может."""


def run_gate(
    device: str = "auto",
    out_dir=None,
    progress: bool = False,
    force: bool = False,
    seed: int = 0,
) -> Dict[str, dict]:
    """Обучить контроль на обоих концах оси ёмкости.

    Возвращает ``{point: record}``.  Кэшируется по ключу запуска, так что второй
    вызов бесплатен.
    """
    records = {}
    for point in GATE_POINTS:
        cfg = gate_config(point, seed=seed)
        if progress:
            print(f"[gate:{point}] {cfg.label()}", flush=True)
        records[point] = run_experiment(
            cfg,
            device=device,
            out_dir=out_dir,
            eval_every=max(1, cfg.steps // 10) if progress else 0,
            progress=progress,
            force=force,
        )
    return records


def _failure_message(point: str, record: dict, threshold: float) -> str:
    cfg = record["config"]
    return (
        f"[{point}] точность на отложенной выборке {record['eval_acc']:.4f} (требуется > {threshold})\n"
        f"    M={cfg['n_pairs']}, D={cfg['n_distractors']}, P={cfg['n_pad']}, "
        f"Q={cfg['n_queries']}, seq_len={record['seq_len']}, W={record['write_count']}\n"
        f"    точность на обучающей выборке {record['train_acc']:.4f}\n"
        f"    {record['in_context_acc']:.4f} = «выдать какое-нибудь значение из контекста» "
        "(вырожденное решение из ловушки T1); "
        f"{record['chance_acc']:.4f} = равномерно по словарю значений"
    )


def require_gate(
    device: str = "auto",
    out_dir=None,
    progress: bool = False,
    force: bool = False,
    seed: int = 0,
    threshold: Optional[float] = None,
) -> Dict[str, dict]:
    """Вернуть записи проверки ``gate`` или бросить :class:`GateFailure`.

    Вызывайте это перед всем, что потребляет результаты перебора.  Это
    исполняемая форма инварианта I6.
    """
    thresh = GATE_THRESHOLD if threshold is None else threshold
    records = run_gate(
        device=device, out_dir=out_dir, progress=progress, force=force, seed=seed
    )
    failures = [p for p, r in records.items() if r["eval_acc"] <= thresh]
    if failures:
        detail = "\n".join(_failure_message(p, records[p], thresh) for p in failures)
        raise GateFailure(
            "softmax-контроль не достиг потолка, поэтому перебор измерял бы "
            "оптимизатор, а не состояние.\n"
            f"{detail}\n"
            "Исправьте гиперпараметры, прежде чем запускать любую сетку (инвариант I6)."
        )
    return records

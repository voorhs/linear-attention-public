"""Крошечная LM, которую обучает эксперимент по ёмкости.

Небольшая модель трансформерной формы с pre-norm, у которой миксер
последовательности сменный.  Четыре ветви делят одну и ту же обвязку слоя:
``softmax`` (контроль, чьё состояние растёт вместе с последовательностью) и
``linear`` / ``gated`` / ``delta``, чьё состояние — фиксированная матрица
``d_k x d_v``.  Три линейных правила вычисляются функцией
:func:`linattn.chunkwise_linear_attention` (часть A плана), так что домашнее
задание не содержит собственной математики и не может разойтись с лестницей
эквивалентностей.

Три проектных решения, которые не косметика:

* ``d_k`` и ``d_v`` независимы везде.  Выходная проекция — это
  ``n_heads*d_v -> d_model``, никогда не ``d_model -> d_model``.  Байты состояния
  растут как ``d_k*d_v``, а предсказанная ёмкость — как ``d_k``, так что
  расширение с уравненными байтами состояния неконструируемо, если эти два
  связаны (раздел 5 спецификации курса).
* **Каузальная depthwise-свёртка** на q, k и v.  Промышленные слои линейного
  внимания её содержат (раздел 3 спецификации курса, блок 3, шаг 4), и она даёт
  сдвиг на предыдущий токен, который нужен схеме ассоциативного воспроизведения:
  без неё модель должна с нуля открыть двухслойную индукционную схему.  Она
  применяется одинаково в каждой ветви, так что не может благоприятствовать одной
  в ущерб другой.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from linattn.chunkwise import chunkwise_linear_attention

from .config import RunConfig


class CausalDepthwiseConv(nn.Module):
    """Depthwise-свёртка по времени, каузальная за счёт паддинга слева."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        t = x.shape[1]
        y = self.conv(x.transpose(1, 2))[..., :t]
        return y.transpose(1, 2)


class SoftmaxMixer(nn.Module):
    """Каузальное softmax-внимание.

    Контрольная ветвь эксперимента: её состояние растёт вместе с
    последовательностью, так что ей не во что упереться по ёмкости.  Если она не
    достигает потолка при наименьшем ``M``, то нет потолка, с которого могли бы
    упасть линейные ветви (инвариант I6).
    """

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_k
        self.d_v = cfg.d_v
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_k, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_k, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_v, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_v, cfg.d_model, bias=False)
        if cfg.short_conv_size > 1:
            self.q_conv = CausalDepthwiseConv(cfg.n_heads * cfg.d_k, cfg.short_conv_size)
            self.k_conv = CausalDepthwiseConv(cfg.n_heads * cfg.d_k, cfg.short_conv_size)
            self.v_conv = CausalDepthwiseConv(cfg.n_heads * cfg.d_v, cfg.short_conv_size)
        else:
            self.q_conv = self.k_conv = self.v_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.q_conv is not None:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)
        q = q.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_v).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(b, t, self.n_heads * self.d_v)
        return self.o_proj(o)


class LinearMixer(nn.Module):
    """Слой линейного внимания: ``linear``, ``gated`` или ``delta``.

    Вся математика целиком — это :func:`linattn.chunkwise_linear_attention`
    (часть A плана, задача A2): дифференцируемый обучающий путь для всех трёх
    правил, вычисляющий ровно то же, что и наивная рекуррентность, при любом
    размере чанка.  Этот класс — только обвязка слоя: проекции, короткая свёртка,
    гейт на каждом шаге времени и нормализация выхода.

    Интерфейс идентичен :class:`SoftmaxMixer` — те же проекции, та же каузальная
    depthwise-свёртка, та же выходная проекция — так что ветви различаются
    правилом обновления и ничем больше.  Три добавления специфичны для линейного
    внимания, и все три — то, что делают промышленные модели (раздел 3
    спецификации курса, блок 3, шаг 4):

    * **q и k L2-нормированы.**  Требуется дельта-правилом, чья рекуррентность
      расходится, как только ``beta_t ||k_t||^2 > 2`` (инвариант I4), и
      стандартно для двух других.  Заметьте: это нормализация *ключей*, а не
      знаменатель ``sum phi(k)``, который раздел 2 спецификации курса запрещает.
    * **RMS-нормализация выхода миксера**, на голову, перед выходной проекцией.
      Считывание линейного внимания — не выпуклая комбинация, и его масштаб
      дрейфует с числом записей; без этого ветвь трудно оптимизировать, а
      недообученная ветвь читается как коллапс ёмкости — ровно тот смешивающий
      фактор, ради разделения которого существует принцип 3 раздела 5
      спецификации курса.  Она масштабирует величину, а не направление, так что
      перекрёстные помехи, которые измеряет эксперимент, остаются нетронутыми.
    * **Смещение (bias) гейта забывания стартует положительным** (ловушка T4).
    """

    def __init__(self, cfg: RunConfig, rule: str) -> None:
        super().__init__()
        if rule not in ("linear", "gated", "delta"):
            raise ValueError(f"неизвестное правило линейного внимания {rule!r}")
        self.rule = rule
        self.chunk_size = cfg.chunk_size
        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_k
        self.d_v = cfg.d_v
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_k, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_k, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_v, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_v, cfg.d_model, bias=False)
        self.o_norm = nn.RMSNorm(cfg.d_v)
        if cfg.short_conv_size > 1:
            self.q_conv = CausalDepthwiseConv(cfg.n_heads * cfg.d_k, cfg.short_conv_size)
            self.k_conv = CausalDepthwiseConv(cfg.n_heads * cfg.d_k, cfg.short_conv_size)
            self.v_conv = CausalDepthwiseConv(cfg.n_heads * cfg.d_v, cfg.short_conv_size)
        else:
            self.q_conv = self.k_conv = self.v_conv = None
        # Один скаляр на голову на шаг времени, ровно как говорит обозначение:
        # alpha_t правила с гейтом и beta_t дельта-правила имеют форму (B, H, N).
        self.gate_proj = (
            nn.Linear(cfg.d_model, cfg.n_heads, bias=True) if rule in ("gated", "delta") else None
        )
        if self.gate_proj is not None:
            nn.init.zeros_(self.gate_proj.weight)
            # Ловушка T4: нулевое смещение даёт sigmoid(0) = 0.5 и коэффициент
            # усиления состояния за чанк 0.5**L, что равно 2.3e-10 при L = 32.
            # Ветвь с гейтом была бы мертва ещё до первого шага и сообщила бы
            # нулевой результат по неверной причине.  beta дельта-правила
            # стартует около 1 по той же причине, но наоборот — оно хочет
            # действительно писать.
            nn.init.constant_(self.gate_proj.bias, cfg.gate_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.q_conv is not None:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)
        q = q.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_v).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        kwargs = {}
        if self.rule == "gated":
            kwargs["decay"] = torch.sigmoid(self.gate_proj(x)).transpose(1, 2)
        elif self.rule == "delta":
            kwargs["beta"] = torch.sigmoid(self.gate_proj(x)).transpose(1, 2)

        o = chunkwise_linear_attention(
            q, k, v, chunk_size=self.chunk_size, rule=self.rule, **kwargs
        )
        o = self.o_norm(o)
        o = o.transpose(1, 2).reshape(b, t, self.n_heads * self.d_v)
        return self.o_proj(o)


def build_mixer(cfg: RunConfig, layer_idx: int) -> nn.Module:
    """Стык между каркасом домашнего задания и эталонными реализациями.

    Возвращает softmax-внимание, когда ``cfg.mixer == "softmax"`` или когда этот
    слой перечислен в ``cfg.full_attention_layers`` (гибридное расширение, которое
    поэтому является изменением конфигурации, а не правкой библиотеки —
    инвариант I8).  Иначе возвращает :class:`LinearMixer`, оборачивающий
    соответствующее правило.
    """
    if cfg.mixer == "softmax" or layer_idx in cfg.full_attention_layers:
        return SoftmaxMixer(cfg)
    return LinearMixer(cfg, cfg.mixer)


class MLP(nn.Module):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden)
        self.fc2 = nn.Linear(hidden, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: RunConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.mixer = build_mixer(cfg, layer_idx)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """Эмбеддинг токенов, обучаемые абсолютные позиции, ``n_layers`` блоков, LM-голова.

    Строится на CPU под явным seed и затем переносится, так что инициализация не
    зависит от устройства (раздел 7 спецификации курса).
    """

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model) if cfg.use_pos_emb else None
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        # Выходы, идущие в остаточную связь (residual), уменьшаются с глубиной —
        # обычное правило GPT-2.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            block.mixer.o_proj.weight.data.mul_(scale)
            block.mlp.fc2.weight.data.mul_(scale)
        # `apply` выше обнуляет смещение каждого Linear, что молча отменило бы
        # положительное смещение гейта забывания и убило бы ветвь с гейтом
        # (ловушка T4).  Восстанавливаем его последним, чтобы ловушку нельзя
        # было вернуть правкой `_init`.
        for block in self.blocks:
            gate_proj = getattr(block.mixer, "gate_proj", None)
            if gate_proj is not None:
                nn.init.zeros_(gate_proj.weight)
                nn.init.constant_(gate_proj.bias, cfg.gate_bias_init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` токены int64 -> ``(B, T, vocab_size)`` логиты."""
        t = tokens.shape[1]
        if t > self.cfg.seq_len:
            raise ValueError(
                f"последовательность длины {t} превышает cfg.seq_len={self.cfg.seq_len}; "
                "seq_len выводится из 2M+D+P+Q и входит в идентичность запуска"
            )
        x = self.tok_emb(tokens)
        if self.pos_emb is not None:
            pos = torch.arange(t, device=tokens.device)
            x = x + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: RunConfig, seed: int) -> TinyLM:
    """Сконструировать на CPU под явным seed.  Вызывающий переносит модель на устройство."""
    gen_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = TinyLM(cfg)
    finally:
        torch.random.set_rng_state(gen_state)
    return model

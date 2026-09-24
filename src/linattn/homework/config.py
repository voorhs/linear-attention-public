"""Конфигурация запуска, идентичность запуска и выбор устройства.

Каждая ручка, которую задание просит студента покрутить, — это поле
:class:`RunConfig`, и оно входит в ключ запуска (инвариант I8 спецификации курса).
Поля, которые меняют *как* запуск выполняется, а не *что* он вычисляет, —
устройство, каталог вывода, частота оценки, печать прогресса — намеренно
**не** являются полями здесь; это аргументы раннера, и только так их можно
по построению держать вне ключа.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Tuple

import torch

# --------------------------------------------------------------------------
# Раскладка словаря.  Фиксирована для каждого запуска в переборе; никогда не
# зависит от M (ловушка T2 спецификации курса).
# --------------------------------------------------------------------------

#: Единственный инертный паддинг-токен.  Один id, один эмбеддинг, так что миксер
#: может научиться ничего для него не записывать.  По теории части 1 его эффект
#: нулевой (ловушка T3).
PAD_ID = 0

#: Позиции без обучающей цели.  Совпадает с ``torch.nn.functional.cross_entropy``.
IGNORE_INDEX = -100

#: Ветви миксера.  В этом раунде реализована только ``softmax``; остальные три
#: поставляются эталонными реализациями (задачи A1/A2 плана).
MIXERS = ("softmax", "linear", "gated", "delta")


class ConfigError(ValueError):
    """Конфигурация, которая не описывает корректно поставленный запуск MQAR."""


@dataclass(frozen=True)
class RunConfig:
    """Одна точка эксперимента по ёмкости.

    Последовательность несёт ``M`` настоящих пар ключ→значение, ``D``
    дистракторных токенов (выдаются как ``D/2`` соседних пар ключ/значение,
    ключи которых никогда не запрашиваются), ``P`` инертных паддинг-токенов и
    ``Q`` токенов-запросов, так что::

        seq_len = 2*M + D + P + Q
        W       = 2*M + D + Q        # записи, стоящие в состоянии на момент запроса

    Выводу части 1 нужна величина ``W`` (ловушка T6), а не ``M``.
    """

    # ---------------- задача ----------------
    n_pairs: int = 4
    """M — настоящие пары ключ→значение.  По два токена и по две записи на каждую."""

    n_distractors: int = 0
    """D — дистракторные **токены** (должно быть чётным; выдаются как D/2 пар).

    Их ключи берутся из того же пула, что и настоящие ключи, и не пересекаются
    с ними, так что на уровне токенов они неотличимы от настоящих пар.  Теория
    предсказывает, что каждый стоит половину сохранённой пары (ловушка T6).
    """

    n_pad: int = 0
    """P — копии :data:`PAD_ID`, вставленные на случайных *границах пар*.

    Никогда между ключом и его значением.  Предсказанный эффект нулевой
    (ловушка T3): это честная форма отрицательного контроля.
    """

    n_queries: int = 4
    """Q — токены-запросы, различные настоящие ключи, выбранные без возвращения."""

    n_key_tokens: int = 256
    n_value_tokens: int = 256

    # ---------------- модель ----------------
    mixer: str = "softmax"
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 1
    d_k: int = 64
    """Размерность головы для q и k.  Не зависит от :attr:`d_v` (S2 спецификации курса)."""
    d_v: int = 64
    """Размерность головы для v и для выхода миксера до выходной проекции."""
    mlp_ratio: int = 4
    short_conv_size: int = 4
    """Ширина каузальной depthwise-свёртки на q, k, v.  0 или 1 отключает её."""
    use_pos_emb: bool = True
    """Обучаемые абсолютные позиционные эмбеддинги.  Поле, а не константа, потому
    что обучаема ли задача без них — это измерение (см. спецификацию раунда 1,
    раздел 1)."""
    chunk_size: int = 32
    """Длина чанка для чанковых линейных ветвей.  Для ``softmax`` не действует,
    но входит в идентичность запуска для каждой ветви, чтобы у перебора
    отрицательного контроля была ручка, которую можно крутить, а его точки
    получали различные ключи запуска (инвариант I8)."""
    gate_bias_init: float = 5.0
    """Начальное смещение (bias) гейта забывания ветви ``gated``, в логит-пространстве.

    Должно быть **положительным** (ловушка T4).  Нулевое смещение даёт
    ``sigmoid(0) = 0.5`` и коэффициент усиления состояния за чанк
    ``0.5**chunk_size`` — 2.3e-10 при ``L = 32`` — так что межчанковое состояние
    уничтожается каждый чанк, а вместе с ним исчезает и градиент вдоль этого пути.
    Ветвь с гейтом тогда сообщает нулевой результат по неверной причине.  При 5.0
    затухание на токен равно 0.993, и состояние сохраняет 42% своей величины на
    протяжении последовательности из 128 токенов.
    """
    full_attention_layers: Tuple[int, ...] = ()
    """Индексы слоёв, принудительно переключённых на softmax независимо от
    :attr:`mixer`.  Гибридное расширение — это изменение конфигурации, а не
    правка библиотеки (инвариант I8)."""

    # ---------------- оптимизация ----------------
    steps: int = 2000
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.1
    warmup_frac: float = 0.1
    final_lr_frac: float = 0.05
    grad_clip: float = 1.0
    n_train: int = 32768
    """Размер *конечной* обучающей выборки.  Конечной намеренно: свежий
    бесконечный поток делает точность на обучающей и на отложенной выборке
    тождественными по построению и разрушает диагностику, которой требует
    принцип 3 раздела S5 спецификации курса."""
    n_eval: int = 2048
    seed: int = 0

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        object.__setattr__(self, "full_attention_layers", tuple(self.full_attention_layers))
        if self.mixer not in MIXERS:
            raise ConfigError(f"mixer должен быть одним из {MIXERS}, получено {self.mixer!r}")
        if self.n_pairs < 1:
            raise ConfigError("n_pairs (M) должно быть >= 1")
        if self.n_distractors < 0 or self.n_distractors % 2:
            raise ConfigError(
                "n_distractors (D) считает дистракторные *токены*, выдаваемые как D/2 "
                f"пар ключ/значение, поэтому должно быть чётным и >= 0; получено {self.n_distractors}"
            )
        if self.n_pad < 0:
            raise ConfigError("n_pad (P) должно быть >= 0")
        if not 1 <= self.n_queries <= self.n_pairs:
            raise ConfigError(
                f"n_queries (Q={self.n_queries}) должно удовлетворять 1 <= Q <= M={self.n_pairs}; "
                "запросы — это различные настоящие ключи, выбранные без возвращения"
            )
        n_distinct = self.n_pairs + self.n_distractors // 2
        if n_distinct > self.n_key_tokens:
            raise ConfigError(
                f"нужно {n_distinct} различных ключей, но пул ключей содержит {self.n_key_tokens}. "
                "Пул намеренно фиксирован на весь перебор (ловушка T2) — поднимайте "
                "n_key_tokens для каждого запуска, а не только для запусков с большим M"
            )
        if n_distinct > self.n_value_tokens:
            raise ConfigError(
                f"нужно {n_distinct} различных значений, но пул значений содержит {self.n_value_tokens}"
            )
        for name in ("d_model", "n_layers", "n_heads", "d_k", "d_v", "mlp_ratio", "chunk_size"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} должно быть >= 1")
        if self.short_conv_size < 0:
            raise ConfigError("short_conv_size должно быть >= 0")
        for idx in self.full_attention_layers:
            if not 0 <= idx < self.n_layers:
                raise ConfigError(
                    f"индекс {idx} в full_attention_layers вне диапазона для n_layers={self.n_layers}"
                )
        if self.steps < 1 or self.batch_size < 1:
            raise ConfigError("steps и batch_size должны быть >= 1")
        if self.n_train < self.batch_size:
            raise ConfigError("n_train должно быть не меньше batch_size")
        if self.n_eval < 1:
            raise ConfigError("n_eval должно быть >= 1")
        if not 0.0 <= self.warmup_frac < 1.0:
            raise ConfigError("warmup_frac должно лежать в [0, 1)")
        if not 0.0 <= self.final_lr_frac <= 1.0:
            raise ConfigError("final_lr_frac должно лежать в [0, 1]")

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Фиксирован на весь перебор — никогда не зависит от M (ловушка T2)."""
        return 1 + self.n_key_tokens + self.n_value_tokens

    @property
    def key_lo(self) -> int:
        return 1

    @property
    def value_lo(self) -> int:
        return 1 + self.n_key_tokens

    @property
    def n_pair_blocks(self) -> int:
        """Настоящие пары плюс пары-дистракторы."""
        return self.n_pairs + self.n_distractors // 2

    @property
    def seq_len(self) -> int:
        return 2 * self.n_pairs + self.n_distractors + self.n_pad + self.n_queries

    @property
    def state_bytes_per_layer(self) -> int:
        """``H * d_k * d_v * dtype_bytes`` — fp32, то есть 4 байта (S1 спецификации курса).

        Имеет смысл только для линейных ветвей; у softmax байты состояния на слой
        равны ``2 * n_heads * d_v * 4 * N``, а его прирост ненулевой.
        """
        return self.n_heads * self.d_k * self.d_v * 4

    # ------------------------------------------------------------------
    def identity(self) -> dict:
        """Поля, которые определяют, *что* вычисляет запуск."""
        out = asdict(self)
        out["full_attention_layers"] = list(self.full_attention_layers)
        return out

    def run_key(self) -> str:
        """Стабильная идентичность этого запуска: 16 шестнадцатеричных символов.

        Не зависит от порядка объявления полей и от всего, что влияет только на
        то, *как* запуск выполняется (устройство, каталог вывода, частота оценки).
        """
        blob = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def replace(self, **kwargs) -> "RunConfig":
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update(kwargs)
        return RunConfig(**data)

    def label(self) -> str:
        """Короткая человекочитаемая метка.  Никогда не используется как
        идентичность — для этого есть :meth:`run_key`."""
        return (
            f"{self.mixer}_M{self.n_pairs}_D{self.n_distractors}_P{self.n_pad}"
            f"_Q{self.n_queries}_dk{self.d_k}_dv{self.d_v}_h{self.n_heads}"
            f"_dm{self.d_model}_L{self.n_layers}_s{self.seed}"
        )


# --------------------------------------------------------------------------
# Выбор устройства (инвариант I9): автоопределение, никогда не cuda по умолчанию.
# --------------------------------------------------------------------------


def resolve_device(name: str = "auto") -> torch.device:
    """``cuda -> mps -> cpu``.

    ``"cuda"`` по умолчанию — это дефект, а не удобство (S6 спецификации курса).
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    """Человекочитаемое имя фактически использованного устройства, для записи о результате."""
    device = torch.device(device)
    if device.type == "cuda":
        try:
            return f"cuda:{torch.cuda.get_device_name(device)}"
        except Exception:  # pragma: no cover - защитная ветка
            return "cuda"
    return device.type


def derive_seed(seed: int, stream: str) -> int:
    """Seed для конкретного потока, выведенный из seed запуска.

    Инициализация модели, обучающая выборка, отложенная выборка и порядок
    минибатчей получают каждый свой поток, так что изменение одного не
    перемешивает молча остальные.
    """
    h = hashlib.sha256(f"{seed}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1)

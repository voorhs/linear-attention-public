"""Каркас (scaffold) домашнего задания: эксперимент по ёмкости линейного внимания.

Всё в этом подпакете работает на CPU. Он никогда не импортирует Triton и никогда
не требует CUDA (инвариант I9 спецификации курса); выбор устройства
автоматический: ``cuda -> mps -> cpu``.

Публичный интерфейс намеренно мал:

- :class:`~linattn.homework.config.RunConfig` — каждая ручка, которую задание
  просит студента покрутить, и ключ запуска, идентифицирующий завершённый запуск.
- :func:`~linattn.homework.data.make_dataset` — генератор MQAR.
- :class:`~linattn.homework.model.TinyLM` — модель со сменным миксером.
- :func:`~linattn.homework.train.run_experiment` — обучить одну конфигурацию.
- :func:`~linattn.homework.gate.require_gate` — блокирующая контрольная проверка
  ``gate`` (инвариант I6).
"""

from .config import IGNORE_INDEX, PAD_ID, RunConfig, resolve_device
from .data import make_dataset, pad_for_seq_len, write_count
from .gate import (
    GATE_CONFIG,
    GATE_THRESHOLD,
    LARGEST_M,
    SMALLEST_M,
    SWEEP_SEQ_LEN,
    GateFailure,
    gate_config,
    require_gate,
    run_gate,
    sweep_config,
)
from .model import LinearMixer, SoftmaxMixer, TinyLM
from .sweep import (
    COLLAPSE_ACC,
    FULL_GRID,
    GRIDS,
    REDUCED_GRID,
    CapacityPoint,
    Grid,
    bisection_runs,
    collapse_point,
    run_grid,
)
from .train import evaluate, run_experiment

__all__ = [
    "COLLAPSE_ACC",
    "FULL_GRID",
    "GATE_CONFIG",
    "GATE_THRESHOLD",
    "IGNORE_INDEX",
    "LARGEST_M",
    "PAD_ID",
    "SMALLEST_M",
    "SWEEP_SEQ_LEN",
    "GRIDS",
    "REDUCED_GRID",
    "CapacityPoint",
    "GateFailure",
    "Grid",
    "LinearMixer",
    "RunConfig",
    "SoftmaxMixer",
    "TinyLM",
    "bisection_runs",
    "collapse_point",
    "evaluate",
    "gate_config",
    "make_dataset",
    "pad_for_seq_len",
    "require_gate",
    "resolve_device",
    "run_experiment",
    "run_gate",
    "run_grid",
    "sweep_config",
    "write_count",
]

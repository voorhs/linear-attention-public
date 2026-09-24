"""Перебор (sweep) по ёмкости: сетки, точка коллапса и возобновляемый раннер.

Вопрос — *сколько ассоциаций «ключ–значение» может удержать слой*, поэтому
величина, которую производит перебор, — это **точка коллапса** ``M*``, одно
число на каждую ``(ветвь, d_k, D, seed)``, плюс каждая пара ``(M, точность)``,
которую пришлось измерить, чтобы её найти.

Два принятых здесь решения — это разница между графиком, который отвечает на
вопрос, и графиком, который фабрикует ответ:

**Точка коллапса определена один раз, операционально.**  ``M*`` — это наибольшее
``M``, такое что точность **на отложенной выборке** не меньше
:data:`COLLAPSE_ACC` = 0.50 в нём *и при каждом меньшем измеренном* ``M``.  На
монотонной кривой это просто «наибольшее ``M``, которое ещё работает»; оговорка
«и при каждом меньшем ``M``» — то, что делает определение корректно поставленным,
когда кривая не монотонна, а такое иногда бывает.  На отложенной выборке, а не на
обучающей, по той же причине, что и проверка ``gate``: модель, которая просто всё
заучила, прошла бы критерий по обучающей выборке.  0.50 намного выше любого
вырожденного решения, которое допускает задача — «выдать какое-нибудь значение
из контекста» набирает ``1/(M + D/2)``, то есть 0.125 внизу оси и 0.031 наверху, —
и намного ниже потолка, так что порог сидит на крутом участке кривой, где он
наименее чувствителен к своему точному значению.  Студентам это определение
даётся, а не предлагается выбрать своё (раздел 5 спецификации курса, рубрика).

**Грубое сканирование, затем бисекция.**  Шаг в 2 раза по обеим осям не может
отличить ``M* ∝ d`` от ``M* ∝ √d``, а считывание ``M*`` по ближайшей точке сетки
квантует его в степени двойки, что фабрикует линейный ответ независимо от истины
(раздел 5 спецификации курса, принцип 4).  Поэтому поиск измеряет ``n_probes``
равномерно расположенных точек, берёт **первое** пересечение порога вниз и
делит отрезок пополам внутри него.

Сканирование — не украшательство.  Голая бисекция вообще не может обнаружить
немонотонную кривую: каждая взятая ею выборка либо поднимает нижнюю границу
отрезка, либо опускает верхнюю, так что множество посещённых ею точек монотонно
*по построению*.  Хуже того, на единственной ветви, которая здесь была измерена
как немонотонная — внимание с гейтом, набравшее 0.9965 при ``M = 8``, 0.0058 при
``M = 16`` и 0.7822 при ``M = 32``, — бисекция, проверившая лишь два конца оси,
нашла бы оба выше порога и сообщила бы, что ветвь **так и не коллапсировала**.
Сканирование видит провал, сообщает ``M*`` в первом пересечении и выставляет
``non_monotone``, чтобы точку можно было исключить или обсудить, а не молча
поверить ей.  Немонотонность — это отказ оптимизации, а не ёмкости.

``M*`` может также выпасть за пределы оси.  ``M`` пробегает ``[Q, 4Q]``, потому
что только там контроль обучаем (раздел 2.2 спецификации), так что ветвь, которая
уже коллапсировала при ``M = Q`` или ещё не коллапсировала к ``M = 4Q``, даёт
**цензурированную** точку.  Цензурированные точки сообщаются как цензурированные
и исключаются из аппроксимации; молчаливое прижатие их к концам оси — это то, как
кривая ёмкости приобретает наклон, которого не заслужила.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import RunConfig
from .data import pad_for_seq_len, write_count
from .gate import (
    LARGEST_D,
    LARGEST_M,
    N_QUERIES,
    SMALLEST_M,
    SWEEP_SEQ_LEN,
    require_gate,
    sweep_config,
)
from .train import run_experiment

#: Операциональное определение точки коллапса.  Одно число, в одном месте, чтобы
#: ``M*`` у каждого студента означало одно и то же.
COLLAPSE_ACC = 0.50


# ---------------------------------------------------------------------------
# Точка коллапса
# ---------------------------------------------------------------------------


@dataclass
class CapacityPoint:
    """Одно измерение точки коллапса."""

    arm: str
    d_k: int
    d_v: int
    n_distractors: int
    seed: int
    chunk_size: int
    m_star: Optional[int]
    censored: Optional[str]
    """``"below"``, если ветвь уже коллапсировала при ``M = SMALLEST_M``,
    ``"above"``, если она не коллапсировала к ``M = LARGEST_M``, иначе ``None``."""
    non_monotone: bool
    evaluated: List[dict]
    n_runs: int
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _point_config(
    arm: str, d_k: int, n_pairs: int, n_distractors: int, seed: int, **overrides
) -> RunConfig:
    return sweep_config(
        n_pairs=n_pairs,
        n_distractors=n_distractors,
        seed=seed,
        mixer=arm,
        d_k=d_k,
        **overrides,
    )


def collapse_point(
    arm: str,
    d_k: int,
    seed: int = 0,
    n_distractors: int = 0,
    resolution: int = 1,
    n_probes: int = 4,
    device: str = "auto",
    out_dir=None,
    progress: bool = False,
    **overrides,
) -> CapacityPoint:
    """Найти ``M*``: грубое сканирование, затем бисекция внутри отрезка, где происходит пересечение.

    Сначала измеряются ``n_probes`` равномерно расположенных точек вдоль оси.
    Они делают три вещи, которые голая бисекция не может: обнажают
    немонотонность, выбирают *первое* пересечение порога вниз, а не то, на
    которое случайно наткнётся бисекция, и образуют кривую точности от ``M``,
    которую отчёт обязан построить.  Затем бисекция уточняет внутри одного
    отрезка, останавливаясь, как только его ширина достигает ``resolution``.
    """
    started = time.perf_counter()
    evaluated: Dict[int, dict] = {}

    def measure(m: int) -> float:
        if m in evaluated:
            return evaluated[m]["eval_acc"]
        cfg = _point_config(arm, d_k, m, n_distractors, seed, **overrides)
        record = run_experiment(cfg, device=device, out_dir=out_dir)
        evaluated[m] = {
            "n_pairs": m,
            "eval_acc": record["eval_acc"],
            "train_acc": record["train_acc"],
            "write_count": record["write_count"],
            "run_key": record["run_key"],
            "seconds": record["seconds"],
        }
        if progress:
            print(
                f"    {arm} d_k={d_k} D={n_distractors} seed={seed} M={m:>3} "
                f"eval={record['eval_acc']:.4f} train={record['train_acc']:.4f}",
                flush=True,
            )
        return record["eval_acc"]

    m_min, m_max = SMALLEST_M, largest_m_for(n_distractors)
    probes = probe_points(m_min, m_max, n_probes)
    probe_acc = [(m, measure(m)) for m in probes]

    censored = None
    m_star: Optional[int] = None
    lo = hi = None
    if probe_acc[0][1] < COLLAPSE_ACC:
        censored = "below"
    elif all(acc >= COLLAPSE_ACC for _, acc in probe_acc):
        censored = "above"
    else:
        # *Первое* пересечение порога вниз, а не последнее.  На немонотонной
        # кривой они различаются, и первое — это то, что имеет в виду
        # определение: наибольшее M, которое ещё работает, прежде чем перестать
        # работать.  Взяв последнее, мы сообщили бы, что ветвь, восстановившаяся
        # наверху оси, так и не коллапсировала — а это ровно измеренный отказ
        # ветви с гейтом.
        for (ma, aa), (mb, ab) in zip(probe_acc, probe_acc[1:]):
            if aa >= COLLAPSE_ACC > ab:
                lo, hi = ma, mb
                break
        while hi - lo > resolution:
            mid = (lo + hi) // 2
            if measure(mid) >= COLLAPSE_ACC:
                lo = mid
            else:
                hi = mid
        m_star = lo

    # Монотонность — допущение поиска, поэтому она проверяется по пробному
    # сканированию, а не принимается на веру.  Собственные выборки бисекции не
    # могут выявить нарушение — каждая выборка либо поднимает нижнюю границу
    # отрезка, либо опускает верхнюю, так что множество посещённых ею точек
    # монотонно по построению; именно поэтому грубое сканирование вообще
    # существует.
    non_monotone = any(
        ab > aa + 0.05 for (_, aa), (_, ab) in zip(probe_acc, probe_acc[1:])
    )
    ordered = [evaluated[m] for m in sorted(evaluated)]

    return CapacityPoint(
        arm=arm,
        d_k=d_k,
        d_v=_point_config(arm, d_k, m_min, n_distractors, seed, **overrides).d_v,
        n_distractors=n_distractors,
        seed=seed,
        chunk_size=_point_config(arm, d_k, m_min, n_distractors, seed, **overrides).chunk_size,
        m_star=m_star,
        censored=censored,
        non_monotone=non_monotone,
        evaluated=ordered,
        n_runs=len(evaluated),
        seconds=time.perf_counter() - started,
    )


def largest_m_for(n_distractors: int) -> int:
    """Верх оси ``M`` при данном числе дистракторов.

    Контроль обучаем, только пока ``M + D/2 <= 4Q`` (раздел 2.2 спецификации),
    так что дистракторы отъедают достижимый диапазон ``M``.  Выход за эту границу
    возвращает запуски на уровне случайного угадывания и для ветви, *и* для
    контроля, а это вообще не измерение ёмкости.
    """
    return min(LARGEST_M, 4 * N_QUERIES - n_distractors // 2)


def bisection_runs(n_distractors: int = 0, resolution: int = 1, n_probes: int = 4) -> int:
    """Число запусков в худшем случае для одного :func:`collapse_point`, для расчёта бюджета.

    Публикуемая стоимость — всегда ``запусков x секунд-на-запуск`` с указанием
    цифры на один запуск (раздел 6 спецификации курса), так что число запусков
    должно быть вычислимо до того, как что-либо запущено.
    """
    probes = probe_points(SMALLEST_M, largest_m_for(n_distractors), n_probes)
    span = max((b - a) for a, b in zip(probes, probes[1:])) if len(probes) > 1 else 0
    steps = 0
    while span > resolution:
        # Шаг бисекции оставляет большую из двух половин, так что ширина
        # сокращается с округлением вверх, а не вниз.  Округление здесь не в ту
        # сторону занижает бюджет на один запуск на точку, а это целый час на
        # сетку.
        span = -(-span // 2)
        steps += 1
    return len(probes) + steps


def probe_points(m_min: int, m_max: int, n_probes: int) -> List[int]:
    """``n_probes`` различных, равномерно расположенных значений ``M``, покрывающих ось."""
    if n_probes < 2:
        raise ValueError("нужны как минимум два конца оси")
    if m_max <= m_min:
        return [m_min]
    raw = [
        m_min + round(i * (m_max - m_min) / (n_probes - 1)) for i in range(n_probes)
    ]
    return sorted(set(raw))


# ---------------------------------------------------------------------------
# Сетки
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Именованный набор измерений и всё, что нужно, чтобы оценить его стоимость."""

    name: str
    capacity_arms: Tuple[str, ...]
    d_ks: Tuple[int, ...]
    seeds: Tuple[int, ...]
    resolution: int
    n_probes: int = 4
    #: Тест на число записей: числа дистракторов при одной фиксированной ветви и ``d_k``.
    distractors: Tuple[int, ...] = ()
    distractor_arm: str = "linear"
    distractor_d_k: int = 32
    distractor_seeds: Tuple[int, ...] = (0,)
    #: Отрицательный контроль: размеры чанка при одной фиксированной ``(ветвь, d_k, M, D)``.
    chunk_sizes: Tuple[int, ...] = ()
    #: Отрицательный контроль: полные длины последовательности, достигаемые инертным паддингом.
    pad_seq_lens: Tuple[int, ...] = ()
    control_arm: str = "linear"
    control_d_k: int = 32
    control_m: int = 16
    #: Панель softmax-контроля: ветвь без предела ёмкости, вдоль всей оси.
    softmax_ms: Tuple[int, ...] = (SMALLEST_M, LARGEST_M)

    def n_runs(self) -> int:
        """Общее число запусков в худшем случае, для публикуемого бюджета ``запусков x на-запуск``."""
        n = len(self.capacity_arms) * len(self.d_ks) * len(self.seeds) * bisection_runs(
            0, self.resolution, self.n_probes
        )
        n += sum(
            bisection_runs(d, self.resolution, self.n_probes) * len(self.distractor_seeds)
            for d in self.distractors
        )
        n += len(self.chunk_sizes) + len(self.pad_seq_lens) + len(self.softmax_ms)
        return n

    def budget(self, seconds_per_run: float) -> dict:
        """``запусков x на-запуск``, никогда не голая сумма (раздел 6 спецификации курса)."""
        runs = self.n_runs()
        return {
            "grid": self.name,
            "runs": runs,
            "seconds_per_run": seconds_per_run,
            "seconds_total": runs * seconds_per_run,
            "hours_total": runs * seconds_per_run / 3600.0,
        }


#: Полная сетка.  Отвечает на весь вопрос: три правила обновления, ось
#: размерности головы, выбранная так, чтобы охватить измеренный перегиб, три
#: seed, точная бисекция, тест на число записей и оба отрицательных контроля.
#:
#: Значения ``d_k`` — не степени двойки, и они малы.  И то и другое — измерения,
#: а не вкус.  ``M`` пробегает ``[Q, 4Q] = [8, 32]``, и точка коллапса
#: пересекает верх этого окна при ``d_k`` где-то около 16-24 для каждой ветви,
#: так что ось размерности головы 32-128 вернула бы почти одни цензурированные
#: точки.  Степени двойки избегаются, потому что считывание ``M*`` по решётке с
#: шагом в 2 раза — это то, что фабрикует линейный ответ (раздел 5 спецификации
#: курса, принцип 4).
FULL_GRID = Grid(
    name="full",
    capacity_arms=("linear", "gated", "delta"),
    d_ks=(6, 8, 10, 12, 14, 16, 20),
    seeds=(0, 1, 2),
    resolution=1,
    n_probes=4,
    distractors=(0, 16, 32),
    distractor_arm="linear",
    distractor_d_k=16,
    distractor_seeds=(0, 1),
    chunk_sizes=(8, 16, 32, 64, 128),
    pad_seq_lens=(64, 96, 128),
    control_arm="linear",
    control_d_k=16,
    control_m=16,
    softmax_ms=(8, 16, 24, 32),
)

#: Сокращённая сетка: та, которую требует задание, размер которой рассчитан из
#: **измеренной** стоимости одного запуска, чтобы её суммарное время на CPU
#: уложилось в ночной прогон.
#:
#: 102 запуска x 248 с/запуск = 7.0 ч на машине из таблицы раздела 16
#: спецификации.  Сумма — это арифметика от измеренной цифры на один запуск, а не
#: измеренная сумма от начала до конца — именно этого просит раздел 6
#: спецификации курса, потому что сумма верна только для тех гиперпараметров и
#: той машины, на которых она была измерена.
#:
#: Что было урезано, в порядке чартера B1.  Сначала ветви: три правила до двух,
#: оставляя ``linear`` (базовую линию, для которой выведена теория) и ``delta``
#: (ту, что по утверждению лекции должна быть лучше), и отбрасывая ``gated`` —
#: ветвь, измеренную как немонотонная и потому наименее информативную на запуск.
#: Затем seed, три до двух, и это нижний предел: один seed — не измерение.  Затем
#: одна точка с оси ``d_k``.
#:
#: ``resolution`` поднимается до 4, а не до 2, и это не стоит ничего реального.
#: Измеренный разброс ``M*`` от seed к seed составляет 2-10 пар (``linear`` при
#: ``d_k = 12`` дал 12 и 22), так что бисекция до +/-2 точнее, чем шум, который
#: она измеряет.  Разрешение 4 выгадывает один запуск на точку ценой точности,
#: которой и так не было.
REDUCED_GRID = Grid(
    name="reduced",
    capacity_arms=("linear", "delta"),
    d_ks=(8, 10, 12, 16),
    seeds=(0, 1),
    resolution=4,
    n_probes=4,
    distractors=(0, 16, 32),
    distractor_arm="linear",
    distractor_d_k=16,
    distractor_seeds=(0,),
    chunk_sizes=(16, 32, 64),
    pad_seq_lens=(64, 128),
    control_arm="linear",
    control_d_k=16,
    control_m=16,
    softmax_ms=(8, 32),
)

GRIDS = {g.name: g for g in (FULL_GRID, REDUCED_GRID)}


# ---------------------------------------------------------------------------
# Раннер
# ---------------------------------------------------------------------------


def _single(cfg: RunConfig, device, out_dir, progress, tag) -> dict:
    record = run_experiment(cfg, device=device, out_dir=out_dir)
    if progress:
        print(
            f"    {tag}  eval={record['eval_acc']:.4f} train={record['train_acc']:.4f}",
            flush=True,
        )
    return {
        "tag": tag,
        "config": record["config"],
        "eval_acc": record["eval_acc"],
        "train_acc": record["train_acc"],
        "seq_len": record["seq_len"],
        "write_count": record["write_count"],
        "state_bytes_per_layer": record["state_bytes_per_layer"],
        "run_key": record["run_key"],
        "seconds": record["seconds"],
    }


def run_grid(
    grid: Grid,
    device: str = "auto",
    out_dir="results/runs",
    summary_dir="results",
    progress: bool = True,
    skip_gate: bool = False,
) -> dict:
    """Выполнить каждое измерение в ``grid`` и записать сводку.

    Возобновляемо с точностью до одного обучающего запуска: каждый запуск
    кэшируется по своему ключу запуска в ``out_dir``, так что прерванная сессия
    стоит не больше запуска, выполнявшегося в этот момент (чартер B1).  Повторный
    запуск всей сетки после сбоя перечитывает кэш и продолжает.

    Проверка ``gate`` идёт первой и не может быть пропущена из командной строки;
    ``skip_gate`` существует только для того, чтобы набор тестов мог прогнать
    раннер, не обучая контроль дважды.
    """
    if not skip_gate:
        require_gate(device=device, out_dir=str(Path(summary_dir) / "gate"), progress=progress)

    started = time.perf_counter()
    summary: dict = {
        "grid": asdict(grid),
        "collapse_acc": COLLAPSE_ACC,
        "capacity": [],
        "write_count_test": [],
        "chunk_size_control": [],
        "padding_control": [],
        "softmax_control": [],
    }

    if progress:
        print(f"[{grid.name}] панель ёмкости: {len(grid.capacity_arms)} ветвей x "
              f"{len(grid.d_ks)} размерностей головы x {len(grid.seeds)} seed", flush=True)
    for arm in grid.capacity_arms:
        for d_k in grid.d_ks:
            for seed in grid.seeds:
                point = collapse_point(
                    arm, d_k, seed=seed, resolution=grid.resolution,
                    n_probes=grid.n_probes, device=device, out_dir=out_dir,
                    progress=progress,
                )
                summary["capacity"].append(point.to_dict())

    if progress and grid.distractors:
        print(f"[{grid.name}] тест на число записей: D в {grid.distractors}", flush=True)
    for n_d in grid.distractors:
        for seed in grid.distractor_seeds:
            point = collapse_point(
                grid.distractor_arm, grid.distractor_d_k, seed=seed, n_distractors=n_d,
                resolution=grid.resolution, n_probes=grid.n_probes, device=device,
                out_dir=out_dir, progress=progress,
            )
            summary["write_count_test"].append(point.to_dict())

    if progress and grid.chunk_sizes:
        print(f"[{grid.name}] отрицательный контроль: размеры чанка {grid.chunk_sizes}", flush=True)
    for chunk in grid.chunk_sizes:
        cfg = _point_config(
            grid.control_arm, grid.control_d_k, grid.control_m, 0, 0, chunk_size=chunk
        )
        summary["chunk_size_control"].append(
            _single(cfg, device, out_dir, progress, f"chunk_size={chunk}")
        )

    if progress and grid.pad_seq_lens:
        print(f"[{grid.name}] отрицательный контроль: паддинг до {grid.pad_seq_lens}", flush=True)
    for seq_len in grid.pad_seq_lens:
        base = _point_config(grid.control_arm, grid.control_d_k, grid.control_m, 0, 0, n_pad=0)
        cfg = base.replace(n_pad=pad_for_seq_len(base, seq_len))
        summary["padding_control"].append(
            _single(cfg, device, out_dir, progress, f"seq_len={seq_len} (P={cfg.n_pad})")
        )

    if progress and grid.softmax_ms:
        print(f"[{grid.name}] панель softmax-контроля: M в {grid.softmax_ms}", flush=True)
    for m in grid.softmax_ms:
        cfg = _point_config("softmax", grid.control_d_k, m, 0, 0)
        summary["softmax_control"].append(_single(cfg, device, out_dir, progress, f"softmax M={m}"))

    summary["seconds"] = time.perf_counter() - started
    summary["n_runs_worst_case"] = grid.n_runs()
    summary["n_runs_actual"] = sum(
        p["n_runs"] for p in summary["capacity"] + summary["write_count_test"]
    ) + len(summary["chunk_size_control"]) + len(summary["padding_control"]) + len(
        summary["softmax_control"]
    )

    if summary_dir is not None:
        path = Path(summary_dir) / f"grid_{grid.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        summary["path"] = str(path)
    return summary

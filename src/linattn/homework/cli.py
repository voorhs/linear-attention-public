"""Точки входа каркаса (scaffold) домашнего задания.

    python -m linattn.homework.cli gate      [--device auto] [--out results/gate]
    python -m linattn.homework.cli time-run  [--device auto] [--arm linear] [--steps N]
    python -m linattn.homework.cli budget    [--grid reduced] --seconds-per-run S
    python -m linattn.homework.cli sweep     [--grid reduced] [--device auto]
    python -m linattn.homework.cli plot      [--grid reduced] [--out figures]

Каждая точка входа принимает явный ``--seed`` и по умолчанию ставит ``--device``
в ``auto`` (``cuda -> mps -> cpu``).  ``"cuda"`` по умолчанию было бы дефектом
(инвариант I9).

``sweep`` вызывает :func:`~linattn.homework.gate.require_gate`, прежде чем
что-либо конструировать.  Нет флага, который пропускает проверку ``gate``.

Из исходного чекаута запускайте всё это с ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import device_name, resolve_device
from .gate import (
    GATE_POINTS,
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
from .sweep import COLLAPSE_ACC, GRIDS, bisection_runs, run_grid
from .train import run_experiment


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps (по умолчанию: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")


def _report(point: str, record: dict) -> None:
    cfg = record["config"]
    print(f"[{point}] {record['label']}")
    print(f"  ключ запуска          : {record['run_key']}")
    print(f"  устройство            : {record['device']}")
    print(
        f"  M={cfg['n_pairs']} D={cfg['n_distractors']} P={cfg['n_pad']} Q={cfg['n_queries']}"
        f"   seq_len={record['seq_len']}   число записей W={record['write_count']}"
    )
    print(f"  параметров            : {record['n_parameters']:,}")
    print(f"  точность на обучающей : {record['train_acc']:.4f}   loss {record['train_loss']:.4f}")
    print(f"  точность на отложенной: {record['eval_acc']:.4f}   loss {record['eval_loss']:.4f}")
    print(
        f"  опорные значения      : {record['in_context_acc']:.4f} «выдать какое-нибудь значение из контекста», "
        f"{record['chance_acc']:.4f} равномерно по словарю значений"
    )
    print(f"  секунды               : {record['seconds']:.1f}  (только для диагностики)")


def cmd_gate(args) -> int:
    print(
        f"Проверка gate (инвариант I6): точность softmax на отложенной выборке > {GATE_THRESHOLD} "
        f"на обоих концах оси ёмкости, M={SMALLEST_M} и M={LARGEST_M}, при seq_len={SWEEP_SEQ_LEN}."
    )
    records = run_gate(
        device=args.device,
        out_dir=args.out,
        progress=not args.quiet,
        force=args.force,
        seed=args.seed,
    )
    passed = True
    for point in GATE_POINTS:
        _report(point, records[point])
        passed = passed and records[point]["eval_acc"] > GATE_THRESHOLD
    print("ПРОВЕРКА GATE ПРОЙДЕНА" if passed else "ПРОВЕРКА GATE НЕ ПРОЙДЕНА")
    return 0 if passed else 1


def cmd_sweep(args) -> int:
    grid = GRIDS[args.grid]
    try:
        records = require_gate(
            device=args.device, out_dir=args.gate_out, progress=not args.quiet, seed=args.seed
        )
    except GateFailure as exc:
        print(str(exc), file=sys.stderr)
        return 2
    accs = ", ".join(f"{p}={records[p]['eval_acc']:.4f}" for p in GATE_POINTS)
    print(f"Проверка gate пройдена ({accs}); контроль достигает потолка на обоих концах оси.\n")
    if args.dry_run:
        b = grid.budget(args.seconds_per_run)
        print(
            f"[{grid.name}] {b['runs']} запусков x {b['seconds_per_run']:.0f} с/запуск "
            f"= {b['hours_total']:.1f} ч. Измерьте свои с/запуск командой `time-run` "
            "и перезапустите это с --seconds-per-run."
        )
        return 0
    summary = run_grid(
        grid,
        device=args.device,
        out_dir=args.out,
        summary_dir=args.summary_dir,
        progress=not args.quiet,
        skip_gate=True,
    )
    print(f"\n[{grid.name}] {summary['n_runs_actual']} обучающих запусков, "
          f"{summary['seconds'] / 3600:.2f} ч по часам на {device_name(resolve_device(args.device))}")
    print(f"  сводка: {summary.get('path')}")
    return 0


def cmd_budget(args) -> int:
    """``запусков x секунд-на-запуск``.  Никогда не голая сумма (раздел 6 спецификации курса)."""
    per_run = args.seconds_per_run
    print(f"Секунд на запуск: {per_run:.0f}  (измерьте командой `time-run` на своей машине)")
    print(f"{'сетка':<10}{'запусков':>9}{'x с/запуск':>12}{'= часов':>10}")
    for name, grid in GRIDS.items():
        if args.grid not in (None, name):
            continue
        b = grid.budget(per_run)
        print(f"{name:<10}{b['runs']:>9}{per_run:>12.0f}{b['hours_total']:>10.2f}")
    print(
        f"\nОдна бисекция для M* стоит не более {bisection_runs(0, 1)} запусков при разрешении 1 "
        f"и {bisection_runs(0, 2)} при разрешении 2."
    )
    return 0


def cmd_time_run(args) -> int:
    """Один представительный запуск, замеренный от начала до конца, на названном устройстве.

    Задача 2 чартера B1 требует, чтобы каждая опубликованная сумма по сетке
    прослеживалась до измерения на один запуск при гиперпараметрах *после
    проверки gate*, поэтому здесь сообщаются секунды на запуск и устройство,
    и никогда — голая сумма.
    """
    if args.arm == "softmax":
        cfg = gate_config(args.point, seed=args.seed)
    else:
        cfg = sweep_config(
            n_pairs=args.n_pairs, seed=args.seed, mixer=args.arm, d_k=args.d_k
        )
    if args.steps is not None:
        cfg = cfg.replace(steps=args.steps)
    dev = resolve_device(args.device)
    print(f"Замер одного запуска на {device_name(dev)}: {cfg.label()}, steps={cfg.steps}")
    record = run_experiment(cfg, device=args.device, out_dir=None, force=True)
    print(f"  секунд/запуск         : {record['seconds']:.1f}")
    print(f"  мс/шаг                : {record['seconds'] / cfg.steps * 1000:.2f}")
    print(f"  устройство            : {record['device']}")
    print(f"  точность на отложенной: {record['eval_acc']:.4f}")
    if args.json:
        print(
            json.dumps(
                {
                    "arm": cfg.mixer,
                    "seconds": record["seconds"],
                    "steps": cfg.steps,
                    "device": record["device"],
                    "eval_acc": record["eval_acc"],
                    "run_key": record["run_key"],
                }
            )
        )
    return 0


def cmd_plot(args) -> int:
    from .plotting import load_summary, plot_all

    path = Path(args.summary_dir) / f"grid_{args.grid}.json"
    if not path.exists():
        print(f"нет сводки сетки по пути {path}; сначала запустите `sweep --grid {args.grid}`", file=sys.stderr)
        return 1
    made = plot_all(load_summary(path), args.out)
    print(f"точка коллапса: наибольшее M с точностью на отложенной выборке >= {COLLAPSE_ACC:.2f}")
    for p in made:
        print(f"  записано {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Полный набор команд, вынесен отдельно, чтобы его можно было проверить без запуска."""
    parser = argparse.ArgumentParser(prog="linattn.homework.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gate", help="запустить блокирующую контрольную проверку gate (инвариант I6)")
    _add_common(p)
    p.add_argument("--out", default="results/gate")
    p.add_argument("--force", action="store_true", help="игнорировать закэшированный результат проверки gate")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("sweep", help="запустить перебор (sweep) по ёмкости (отказывает, если проверка gate не пройдена)")
    _add_common(p)
    p.add_argument("--grid", default="reduced", choices=sorted(GRIDS))
    p.add_argument("--out", default="results/runs")
    p.add_argument("--summary-dir", default="results")
    p.add_argument("--gate-out", default="results/gate")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="выполнить проверку gate и оценить стоимость сетки, затем остановиться, не обучая её",
    )
    p.add_argument("--seconds-per-run", type=float, default=300.0)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("budget", help="запусков x секунд-на-запуск для каждой сетки")
    p.add_argument("--grid", default=None, choices=sorted(GRIDS))
    p.add_argument("--seconds-per-run", type=float, required=True)
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("time-run", help="замерить один представительный запуск на этом устройстве")
    _add_common(p)
    p.add_argument("--arm", default="softmax", choices=("softmax", "linear", "gated", "delta"))
    p.add_argument("--point", default="smallest_m", choices=GATE_POINTS)
    p.add_argument("--n-pairs", type=int, default=SMALLEST_M, dest="n_pairs")
    p.add_argument("--d-k", type=int, default=32, dest="d_k")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_time_run)

    p = sub.add_parser("plot", help="построить графики для отчёта по сводке сетки")
    p.add_argument("--grid", default="reduced", choices=sorted(GRIDS))
    p.add_argument("--summary-dir", default="results")
    p.add_argument("--out", default="figures")
    p.set_defaults(func=cmd_plot)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Plotting helpers for the capacity experiment.

Scaffolding, not graded work (course spec section 5, principle 6).  What is *not*
here is the threshold fit: locating ``M*`` is a definition and lives in
:mod:`linattn.homework.sweep`, but fitting ``M* = c·d_k + b`` and comparing ``c``
to the Part 1 prediction is the student's own code, and the budget in the
assignment text accounts for writing it.

``matplotlib`` is an optional extra (``pip install -e ".[plot]"``), so it is
imported inside the functions: a student who only wants the numbers never needs
it, and the homework's base install stays small.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .sweep import COLLAPSE_ACC

_ARM_ORDER = ("softmax", "linear", "gated", "delta")


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "plotting needs matplotlib, which is an optional extra: "
            'pip install -e ".[plot]"'
        ) from exc
    return plt


def load_summary(path) -> dict:
    return json.loads(Path(path).read_text())


def plot_accuracy_vs_m(summary: dict, path, title: Optional[str] = None) -> str:
    """Accuracy against ``M``, one line per ``(arm, d_k, seed)``, with the 0.50 rule drawn.

    Both train and held-out accuracy are drawn, dashed and solid: a capacity
    ceiling collapses both together, while a gap between them means the arm did
    not fit its training set and the point is measuring the optimizer (course
    spec section 5, principle 3).
    """
    plt = _require_matplotlib()
    points = summary["capacity"]
    fig, axes = plt.subplots(
        1, max(1, len({p["arm"] for p in points})), figsize=(4.2 * max(1, len({p["arm"] for p in points})), 3.6),
        sharey=True, squeeze=False,
    )
    arms = [a for a in _ARM_ORDER if a in {p["arm"] for p in points}]
    for ax, arm in zip(axes[0], arms):
        for i, d_k in enumerate(sorted({p["d_k"] for p in points if p["arm"] == arm})):
            colour = f"C{i}"
            labelled = False
            # One line per *seed*.  Joining points from different seeds would draw
            # the seed spread as a sawtooth in M, which is exactly the reading
            # error the separate-seeds rule exists to prevent.
            for p in points:
                if p["arm"] != arm or p["d_k"] != d_k:
                    continue
                ev = sorted(p["evaluated"], key=lambda e: e["n_pairs"])
                xs = [e["n_pairs"] for e in ev]
                ax.plot(
                    xs, [e["eval_acc"] for e in ev], marker="o", ms=3, color=colour,
                    label=None if labelled else f"d_k={d_k}",
                )
                ax.plot(
                    xs, [e["train_acc"] for e in ev], linestyle="--", lw=0.8, color=colour,
                )
                labelled = True
        ax.axhline(COLLAPSE_ACC, color="0.4", lw=0.8, ls=":")
        ax.set_title(arm)
        ax.set_xlabel("M (key-value pairs)")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("accuracy (solid: held-out, dashed: train)")
    axes[0][-1].legend(fontsize=7, ncol=2)
    fig.suptitle(title or "Associative recall vs number of stored pairs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_collapse_vs_dk(summary: dict, path, title: Optional[str] = None) -> str:
    """``M*`` against ``d_k``, one series per arm, seeds shown individually.

    Censored points -- an arm already collapsed at the bottom of the axis, or not
    yet collapsed at the top -- are drawn as open markers at the axis edge and
    must be excluded from any fit.  Drawing them is deliberate: a reader has to be
    able to see how much of the curve was actually measured.
    """
    plt = _require_matplotlib()
    points = summary["capacity"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    from .gate import LARGEST_M, SMALLEST_M

    for arm in [a for a in _ARM_ORDER if a in {p["arm"] for p in points}]:
        xs = [p["d_k"] for p in points if p["arm"] == arm and p["m_star"] is not None]
        ys = [p["m_star"] for p in points if p["arm"] == arm and p["m_star"] is not None]
        sc = ax.scatter(xs, ys, s=22, label=arm)
        cx = [p["d_k"] for p in points if p["arm"] == arm and p["censored"]]
        cy = [
            SMALLEST_M if p["censored"] == "below" else LARGEST_M
            for p in points
            if p["arm"] == arm and p["censored"]
        ]
        ax.scatter(cx, cy, s=26, facecolors="none", edgecolors=sc.get_facecolor(), marker="^")
    ax.axhspan(0, SMALLEST_M, color="0.9", zorder=0)
    ax.axhspan(LARGEST_M, LARGEST_M * 1.5, color="0.9", zorder=0)
    ax.set_ylim(0, LARGEST_M * 1.2)
    ax.set_xlabel("d_k (head dimension)")
    ax.set_ylabel(f"M*  (largest M with held-out accuracy >= {COLLAPSE_ACC:.2f})")
    ax.set_title(title or "Collapse point vs head dimension")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_write_count_test(summary: dict, path, title: Optional[str] = None) -> str:
    """``M*`` against distractor count ``D``, with the predicted ``-D/2`` line.

    The prediction being tested is that written filler consumes capacity at half
    a stored pair each: `M*(D) = M*(0) - D/2`.  The two ways to get it wrong are
    drawn as well -- no shift at all (the naive "capacity is independent of
    sequence length" reading) and a shift of `-D` (writes miscounted as pairs).
    """
    plt = _require_matplotlib()
    points = summary["write_count_test"]
    if not points:
        raise ValueError("this summary has no write-count panel")
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    xs = [p["n_distractors"] for p in points]
    ys = [p["m_star"] for p in points]
    ax.scatter(xs, ys, s=26, color="C0", label="measured M*", zorder=3)
    base = next((p["m_star"] for p in points if p["n_distractors"] == 0), None)
    if base is not None:
        ds = sorted(set(xs))
        ax.plot(ds, [base for _ in ds], ls=":", color="0.5", label="no shift (wrong)")
        ax.plot(ds, [base - d / 2 for d in ds], ls="-", color="C1", label="predicted -D/2")
        ax.plot(ds, [base - d for d in ds], ls="--", color="0.5", label="-D (writes as pairs)")
    ax.set_xlabel("D (written distractor tokens)")
    ax.set_ylabel("M*")
    ax.set_title(title or "Write-count test: W = 2M + D + Q")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_negative_controls(summary: dict, path, title: Optional[str] = None) -> str:
    """The two nulls: accuracy against chunk size, and against sequence length.

    Both are predicted flat, for different reasons, so both failing the same way
    would mean something quite different from either failing alone.
    """
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    chunk = summary.get("chunk_size_control", [])
    if chunk:
        xs = [c["config"]["chunk_size"] for c in chunk]
        axes[0].plot(xs, [c["eval_acc"] for c in chunk], marker="o")
        axes[0].plot(xs, [c["train_acc"] for c in chunk], marker="o", ls="--", lw=0.8)
        axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("chunk size L")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("chunk size: exact null (I2)")
    pad = summary.get("padding_control", [])
    if pad:
        axes[1].plot([c["seq_len"] for c in pad], [c["eval_acc"] for c in pad], marker="o")
        axes[1].plot(
            [c["seq_len"] for c in pad], [c["train_acc"] for c in pad], marker="o", ls="--", lw=0.8
        )
    axes[1].set_xlabel("sequence length N (inert padding)")
    axes[1].set_title("padding: predicted null")
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    fig.suptitle(title or "Negative controls (solid: held-out, dashed: train)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_all(summary: dict, out_dir) -> list:
    """Every figure the report needs, from one grid summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = [
        plot_accuracy_vs_m(summary, out / "accuracy-vs-m.png"),
        plot_collapse_vs_dk(summary, out / "collapse-vs-dk.png"),
    ]
    if summary.get("write_count_test"):
        made.append(plot_write_count_test(summary, out / "write-count-test.png"))
    if summary.get("chunk_size_control") or summary.get("padding_control"):
        made.append(plot_negative_controls(summary, out / "negative-controls.png"))
    return made

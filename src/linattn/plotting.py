"""Figures for the seminar (plan Task A4).

Two rules shape everything here.

**No display is required.**  ``matplotlib`` is imported lazily and, when nothing
else has chosen a backend, Agg is selected before ``pyplot`` first loads -- so a
headless box, a CI runner, and a Kaggle kernel all render.  If the notebook has
already imported ``pyplot`` (``%matplotlib inline``), its choice is left alone.

**A series that could not be measured is omitted and labelled, never faked.**
Peak memory has no API outside CUDA (trap T10).  Plotting an all-NaN series into
a log-scaled axis produces an empty panel that reads as "no memory used", which
is worse than no panel at all: it is a wrong answer rendered confidently.  So
:func:`plot_sweep` draws nothing in that panel and prints the reason across it.
Points a sweep skipped because they did not fit (trap T9) are likewise absent
from the line -- a gap -- and named in the caption underneath.

``matplotlib`` is an optional dependency (``pip install linattn[plot]``); nothing
in :mod:`linattn.bench` imports it.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Sequence

from linattn.bench import (
    LinearAttentionLayer,
    SoftmaxAttentionLayer,
    SweepResult,
    format_bytes,
)

__all__ = [
    "plot_peak_memory",
    "plot_state_bytes",
    "plot_sweep",
    "plot_times",
]

_AXIS_LABELS: Mapping[str, str] = {
    "n": "sequence length N",
    "chunk_size": "chunk size L",
    "head_dims": "(d_k, d_v)",
}


def _pyplot():
    """``matplotlib.pyplot``, with a non-interactive backend when there is no display.

    The backend is only forced when ``pyplot`` has not been imported yet and the
    user has not set ``MPLBACKEND``: a seminar notebook that already ran
    ``%matplotlib inline`` keeps its inline backend and still renders.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'linattn[plot]'"
        ) from exc
    if "matplotlib.pyplot" not in sys.modules and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path) -> Any:
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def _skipped_caption(result: SweepResult) -> str | None:
    skipped = result.skipped
    if not skipped:
        return None
    parts = [
        f"{p.key.form} at {result.axis_label(p)} "
        f"(predicted {format_bytes(p.predicted_bytes)})"
        for p in skipped
    ]
    return "skipped, did not fit: " + "; ".join(parts)


def _series(result: SweepResult, form: str, attribute: str):
    """``(xs, ys, tick_labels)`` for one form, with skipped points left out."""
    xs: list[Any] = []
    ys: list[float] = []
    labels: list[str] = []
    for index, point in enumerate(result.for_form(form)):
        value = getattr(point, attribute)
        if value is None:
            continue  # a gap: skipped, planned, or unmeasurable
        raw = result.axis_value(point)
        xs.append(raw if isinstance(raw, int) else index)
        ys.append(float(value))
        labels.append(result.axis_label(point))
    return xs, ys, labels


def _categorical(result: SweepResult) -> bool:
    return result.axis == "head_dims"


def _draw_panel(ax, result: SweepResult, attribute: str, ylabel: str) -> int:
    drawn = 0
    tick_labels: list[str] = []
    for form in result.forms:
        xs, ys, labels = _series(result, form, attribute)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=form)
        drawn += 1
        if len(labels) > len(tick_labels):
            tick_labels = labels
    ax.set_xlabel(_AXIS_LABELS.get(result.axis, result.axis))
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    if drawn:
        ax.legend(frameon=False, fontsize="small")
        ax.set_yscale("log")  # only ever with positive measured data
        if _categorical(result):
            ax.set_xticks(range(len(tick_labels)), tick_labels)
        else:
            ax.set_xscale("log", base=2)
    return drawn


def _memory_note(result: SweepResult) -> str:
    for point in result.points:
        if point.memory is not None and point.memory.reason:
            return point.memory.reason
    if result.dry_run:
        return "peak memory not measured: this was a dry run"
    device = result.points[0].key.device if result.points else "cpu"
    return f"peak memory not measured on {device}: no point in this sweep ran"


def plot_times(result: SweepResult, *, ax=None, path=None):
    """Wall-clock time against the swept axis, one line per form."""
    plt = _pyplot()
    fig = ax.figure if ax is not None else plt.figure(figsize=(6.0, 4.2))
    if ax is None:
        ax = fig.add_subplot(111)
    _draw_panel(ax, result, "seconds", "time per call (s)")
    ax.set_title("wall clock", fontsize="medium")
    return _finish(fig, path)


def plot_peak_memory(result: SweepResult, *, ax=None, path=None):
    """Peak memory against the swept axis -- or the reason there is no series.

    Where the device has no peak-memory API this panel is deliberately empty of
    data and carries the reason instead (trap T10).
    """
    plt = _pyplot()
    fig = ax.figure if ax is not None else plt.figure(figsize=(6.0, 4.2))
    if ax is None:
        ax = fig.add_subplot(111)
    drawn = _draw_panel(ax, result, "peak_bytes", "peak memory (bytes)")
    ax.set_title("peak memory", fontsize="medium")
    if not drawn:
        ax.text(
            0.5,
            0.5,
            _wrap(_memory_note(result), 34),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize="small",
            color="0.25",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    return _finish(fig, path)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def plot_sweep(result: SweepResult, *, path=None, title: str | None = None):
    """The seminar's two-panel benchmark figure: wall clock and peak memory.

    The suptitle carries :meth:`linattn.bench.SweepResult.describe`, so the
    figure states its ``B`` and ``H`` (trap T9) and both head dimensions
    (spec I8) rather than leaving a reader to guess what was held fixed.  Points
    the sweep skipped appear as a gap in the line and are named in the caption.
    """
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    plot_times(result, ax=axes[0])
    plot_peak_memory(result, ax=axes[1])

    heading = title or "linear attention: three views, one function"
    fig.suptitle(f"{heading}\n{result.describe()}", fontsize="medium")

    caption = _skipped_caption(result)
    if caption:
        fig.text(0.5, -0.04, caption, ha="center", fontsize="small", color="0.25")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish(fig, path)


def plot_state_bytes(
    lengths: Sequence[int] = (4096, 32768, 131072),
    layers: Mapping[str, Any] | None = None,
    *,
    path=None,
    title: str | None = None,
):
    """Per-layer state bytes against context length, with each marginal in the legend.

    Both of spec section 1's numbers appear: the curve is **per-layer state bytes
    at context length N**, and every legend entry names that layer's **marginal
    bytes per token, per layer** -- constant and non-zero for softmax attention,
    exactly zero for every linear-attention layer.  Two numbers, two names, never
    merged into one label.

    The default is spec section 1's reference model: a Llama-3-8B-shaped softmax
    layer (``n_kv_heads=8``, ``head_dim=128``, fp16) against a linear layer of
    the same width (``H=32``, ``d_k=d_v=128``, fp16).
    """
    plt = _pyplot()
    if layers is None:
        layers = {
            "softmax, Llama-3-8B-shaped (n_kv_heads=8, head_dim=128, fp16)":
                SoftmaxAttentionLayer(n_kv_heads=8, head_dim=128, dtype_bytes=2),
            "linear, same width (H=32, d_k=128, d_v=128, fp16)":
                LinearAttentionLayer(heads=32, d_k=128, d_v=128, dtype_bytes=2),
        }

    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_subplot(111)
    for name, layer in layers.items():
        marginal = layer.marginal_bytes_per_token()
        label = f"{name} -- marginal {format_bytes(marginal)}/token/layer"
        ax.plot(
            list(lengths),
            [layer.state_bytes(n) for n in lengths],
            marker="o",
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length N")
    ax.set_ylabel("per-layer state bytes at N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize="small", loc="best")
    ax.set_title(title or "the two numbers, per layer", fontsize="medium")
    fig.tight_layout()
    return _finish(fig, path)

"""Timing, peak-memory, and sweep helpers for the seminar (plan Task A4).

Everything here is *diagnostic*.  No number this module produces is a
correctness criterion, no test asserts on a timing value (spec section 6), and
nothing in the homework depends on a measurement taken here (spec I9).  What the
module is for is the seminar's central demonstration: the form with fewer FLOPs
is the slower one, and the finite state is the reason decode goes flat.

Four things are load-bearing, each of them a trap paid for once already.

**It synchronizes the device actually in use (trap T10).**  A helper that flushes
only CUDA reports fiction elsewhere: measured on MPS, 0.000091 s reported
against 0.112 s actual -- an understatement of three orders of magnitude,
silently, producing a plausible and entirely false curve.  :func:`synchronize`
dispatches on ``torch.device(...).type`` and warns rather than lying when a
backend exposes no flush at all.

**Peak memory has no API outside CUDA.**  ``torch.mps`` offers
``current_allocated_memory`` and ``driver_allocated_memory`` but no peak, and
CPU offers nothing that measures one call rather than the whole process.  So the
series is *omitted* where it cannot be measured -- :class:`MemoryResult` carries
``peak_bytes = None`` and a reason, never a NaN.  An all-NaN series drawn into a
log axis renders as an empty panel and reads as "no memory used", which is the
defect, not the fix.  :func:`linattn.plotting.plot_sweep` prints the reason on
the panel instead.

**A sweep over ``N`` states its ``B`` and ``H`` and skips what it cannot fit
(trap T9).**  The naive parallel form holds two ``(B, H, N, N)`` tensors at once;
at ``B*H = 8, N = 16384`` in fp32 that is exactly 16 GiB, which OOMs a T4 -- the
reference platform, not a small card.  Every point is priced by
:func:`predicted_peak_bytes` against :func:`device_memory_budget` *before* it
runs, and a point that does not fit becomes a recorded gap in the table and on
the plot rather than a crash.

**``d_k`` and ``d_v`` are separate parameters and separate key fields (spec I8,
section 2).**  The homework's state-bytes-matched extension compares
``(d_k=64, d_v=16)`` against ``(d_k=32, d_v=32)`` at equal state bytes; a scaffold
that ties them cannot express it.  :class:`RunKey` records both, and
:func:`sweep_head_dims` sweeps them.

Also here: :func:`prefill_state`, which takes the state from A2's chunkwise pass
instead of re-deriving it with a per-token Python loop (trap T11 -- at a 32k
context that loop is 32768 interpreter iterations per measured point), and the
two numbers of spec section 1, :class:`LinearAttentionLayer` and
:class:`SoftmaxAttentionLayer`, whose arithmetic matches ``scripts/state_bytes.py``
exactly.

Everything runs on CPU, so the notebook is developable on a laptop.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, NamedTuple, Sequence

import torch
from torch import Tensor

from linattn.chunkwise import DEFAULT_CHUNK_SIZE, chunkwise_linear_attention
from linattn.reference import (
    RULES,
    Rule,
    parallel_linear_attention,
    recurrent_linear_attention,
)

__all__ = [
    "DEFAULT_ITERS",
    "DEFAULT_MEMORY_FRACTION",
    "DEFAULT_WARMUP",
    "FORMS",
    "T4_TOTAL_BYTES",
    "BenchForm",
    "BenchInputs",
    "BenchWarning",
    "LinearAttentionLayer",
    "MemoryBudget",
    "MemoryResult",
    "RunKey",
    "SoftmaxAttentionLayer",
    "SweepPoint",
    "SweepResult",
    "TimingResult",
    "device_memory_budget",
    "device_synchronizer",
    "dtype_bytes",
    "format_bytes",
    "get_form",
    "make_bench_inputs",
    "marginal_bytes_per_token",
    "measure_peak_memory",
    "peak_memory_supported",
    "predicted_peak_bytes",
    "prefill_state",
    "register_form",
    "resolve_device",
    "run_sweep",
    "state_bytes",
    "sweep_chunk_size",
    "sweep_context_length",
    "sweep_head_dims",
    "synchronize",
    "time_callable",
]

LOGGER = logging.getLogger("linattn.bench")

#: Small on purpose: these cells run in front of an audience (Charter B2), and a
#: benchmark that takes a minute per point is a benchmark nobody watches.
DEFAULT_WARMUP = 1
DEFAULT_ITERS = 3

#: How much of the device's free memory a sweep is willing to reach for.  The
#: rest is headroom for allocator fragmentation and for whatever else the
#: notebook is holding.
DEFAULT_MEMORY_FRACTION = 0.8

#: Nameplate memory of the reference GPU (spec section 6: a single NVIDIA T4,
#: ``sm_75``, 16 GB).  Usable memory on a real T4 is lower, which only makes a
#: skip decision taken against this number more conservative, never less.  It is
#: here so a sweep can be *planned* for the class machine from a dev laptop.
T4_TOTAL_BYTES = 16 * 1024**3


class BenchWarning(UserWarning):
    """Something a measurement cannot do, said out loud instead of faked."""


# --------------------------------------------------------------------------- #
# devices and synchronization (trap T10)
# --------------------------------------------------------------------------- #
#: Device types whose work is already complete when the Python call returns.
SYNCHRONOUS_DEVICE_TYPES = frozenset({"cpu", "meta"})


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """``cuda -> mps -> cpu``, in that order, unless the caller names one.

    A default of ``"cuda"`` is a defect, not a convenience (spec I9): the
    homework runs on whatever hardware a student owns and the notebook is
    developed on a laptop.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_synchronizer(
    device: str | torch.device,
) -> Callable[[], None] | None:
    """The callable that flushes ``device``'s queue, or ``None`` if it has none.

    Dispatch is on the device *type*, resolved through ``torch``'s own backend
    module (``torch.cuda``, ``torch.mps``, ``torch.xpu``, ...), so a device this
    library has never heard of works the moment torch grows a backend for it.
    Hardcoding CUDA here is exactly trap T10, and
    ``tests/test_bench.py::test_synchronize_dispatches_to_the_backend_of_the_device_type``
    exists to catch a future edit that does.

    Returns ``None`` for CPU, where there is nothing to flush.  Warns and returns
    ``None`` for an accelerator whose backend module exposes no ``synchronize``:
    the timing that follows is then understated, and the caller is told so rather
    than handed a beautiful false curve.
    """
    device_type = torch.device(device).type
    if device_type in SYNCHRONOUS_DEVICE_TYPES:
        return None
    backend = getattr(torch, device_type, None)
    sync = getattr(backend, "synchronize", None)
    if sync is None or not callable(sync):
        warnings.warn(
            f"device type {device_type!r} exposes no torch.{device_type}.synchronize(), "
            f"so timings on it are the launch cost and not the work (trap T10)",
            BenchWarning,
            stacklevel=2,
        )
        return None
    return sync


def synchronize(device: str | torch.device) -> None:
    """Block until ``device`` has finished the work already submitted to it."""
    sync = device_synchronizer(device)
    if sync is not None:
        sync()


def dtype_bytes(dtype: torch.dtype | str) -> int:
    """Bytes per element of ``dtype``, accepted as a dtype or as its name."""
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"unknown dtype name {dtype!r}")
        dtype = resolved
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    return torch.empty((), dtype=dtype).element_size()


def _dtype_name(dtype: torch.dtype | str) -> str:
    return dtype if isinstance(dtype, str) else str(dtype).removeprefix("torch.")


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimingResult:
    """One timed measurement, with the samples it was reduced from.

    ``seconds`` is the reduction (median by default) over ``iters`` measured
    calls, each bracketed by a flush of the device in use.  It is a diagnostic
    number: nothing asserts on it.
    """

    seconds: float
    samples: tuple[float, ...]
    warmup: int
    iters: int
    device: str
    synchronized: bool
    reduce: str = "median"
    label: str | None = None

    @property
    def best(self) -> float:
        return min(self.samples)

    @property
    def worst(self) -> float:
        return max(self.samples)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.seconds * 1e3:.3f} ms ({self.reduce} of {self.iters} on {self.device})"


_REDUCERS: Mapping[str, Callable[[Sequence[float]], float]] = {
    "median": statistics.median,
    "mean": statistics.fmean,
    "min": min,
}


def time_callable(
    fn: Callable[[], Any],
    *,
    device: str | torch.device | None = None,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    reduce: str = "median",
    label: str | None = None,
) -> TimingResult:
    """Time ``fn`` on ``device``, flushing **that** device around every call.

    Args:
        fn: zero-argument callable running exactly one iteration of the work.
        device: the device the work runs on.  Defaults to
            :func:`resolve_device`.  This is what gets synchronized, and getting
            it wrong is trap T10, not a rounding error.
        warmup: untimed calls first (allocator warm, kernels autotuned, lazy
            module imports done).  Small by default: these run in class.
        iters: timed calls.  Also small by default.
        reduce: ``"median"`` (default), ``"mean"``, or ``"min"``.
        label: free text carried through to the result.

    Returns:
        A :class:`TimingResult`.  Every value in it is diagnostic.
    """
    if warmup < 0:
        raise ValueError(f"`warmup` must be >= 0; got {warmup}")
    if iters < 1:
        raise ValueError(f"`iters` must be >= 1; got {iters}")
    if reduce not in _REDUCERS:
        raise ValueError(f"unknown `reduce` {reduce!r}; expected one of {sorted(_REDUCERS)}")

    torch_device = resolve_device(device)
    sync = device_synchronizer(torch_device)

    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    samples: list[float] = []
    for _ in range(iters):
        if sync is not None:
            sync()  # do not charge this iteration for the previous one's tail
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()  # the work is not done until the device says it is
        samples.append(time.perf_counter() - start)

    return TimingResult(
        seconds=float(_REDUCERS[reduce](samples)),
        samples=tuple(samples),
        warmup=warmup,
        iters=iters,
        device=str(torch_device),
        synchronized=sync is not None,
        reduce=reduce,
        label=label,
    )


# --------------------------------------------------------------------------- #
# peak memory -- measured where an API exists, omitted and explained where not
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemoryResult:
    """Peak allocator-visible memory during one call, or the reason there is none.

    ``peak_bytes`` is ``None`` -- never ``NaN``, never ``0.0`` -- when the device
    exposes no peak-memory API.  Consumers must render that absence visibly
    (trap T10); :mod:`linattn.plotting` omits the series and prints ``reason``
    on the panel.
    """

    device: str
    available: bool
    peak_bytes: int | None = None
    baseline_bytes: int | None = None
    reason: str | None = None

    @property
    def delta_bytes(self) -> int | None:
        """Peak above what was already allocated when the call started."""
        if self.peak_bytes is None or self.baseline_bytes is None:
            return None
        return max(self.peak_bytes - self.baseline_bytes, 0)


def _peak_memory_backend(device: str | torch.device):
    """``(backend_module, reset, read)`` for ``device``, or ``None``."""
    device_type = torch.device(device).type
    backend = getattr(torch, device_type, None)
    reset = getattr(backend, "reset_peak_memory_stats", None)
    read = getattr(backend, "max_memory_allocated", None)
    current = getattr(backend, "memory_allocated", None)
    if callable(reset) and callable(read):
        return backend, reset, read, current
    return None


def peak_memory_supported(device: str | torch.device) -> bool:
    """Whether ``device`` can report a peak for a single call.

    True on CUDA (and any other torch backend that grew the same two functions).
    False on CPU and MPS: ``torch.mps`` reports current and driver-allocated
    memory but no peak, and the CPU allocator reports nothing per call at all.
    A process-RSS sampler is not a substitute -- glibc does not return freed
    arenas, so the number it produces is the process high-water mark and not
    this call's footprint.
    """
    return _peak_memory_backend(device) is not None


def measure_peak_memory(
    fn: Callable[[], Any], *, device: str | torch.device | None = None
) -> tuple[Any, MemoryResult]:
    """Run ``fn`` once and report the peak memory it reached.

    Returns ``(value, MemoryResult)``.  On a device with no peak-memory API the
    call still runs and the result carries ``peak_bytes=None`` plus the reason,
    so a caller can print the gap instead of inventing a number for it.
    """
    torch_device = resolve_device(device)
    backend = _peak_memory_backend(torch_device)
    if backend is None:
        value = fn()
        return value, MemoryResult(
            device=str(torch_device),
            available=False,
            peak_bytes=None,
            reason=(
                f"no peak-memory API on device type {torch_device.type!r}: the series "
                f"is omitted rather than plotted as zeros (trap T10)"
            ),
        )

    _, reset, read, current = backend
    synchronize(torch_device)
    baseline = int(current(torch_device)) if callable(current) else None
    reset(torch_device)
    value = fn()
    synchronize(torch_device)
    peak = int(read(torch_device))
    return value, MemoryResult(
        device=str(torch_device),
        available=True,
        peak_bytes=peak,
        baseline_bytes=baseline,
    )


# --------------------------------------------------------------------------- #
# how much memory is there, and how much would this point need (trap T9)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemoryBudget:
    """What a sweep is willing to allocate on a device, and where that came from.

    ``usable_bytes is None`` means the budget could not be determined; a sweep
    then skips nothing and says so, because skipping everything on an unknown
    budget is worse than running and letting a real OOM be caught.
    """

    device: str
    usable_bytes: int | None
    total_bytes: int | None
    free_bytes: int | None
    fraction: float
    source: str

    def fits(self, nbytes: int) -> bool:
        return self.usable_bytes is None or nbytes <= self.usable_bytes

    def describe(self) -> str:
        if self.usable_bytes is None:
            return f"budget unknown ({self.source})"
        return f"{format_bytes(self.usable_bytes)} usable ({self.source})"


def _host_memory_bytes() -> tuple[int | None, str]:
    """Available host RAM, and where the number came from.

    ``/proc/meminfo``'s ``MemAvailable`` is the honest Linux number -- it counts
    reclaimable page cache, which ``SC_AVPHYS_PAGES`` does not, and on this
    machine the two differ by 20x.  macOS has no ``/proc``, so ``sysconf`` is the
    fallback there.
    """
    meminfo = "/proc/meminfo"
    try:
        with open(meminfo, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024, f"{meminfo} MemAvailable"
    except OSError:
        pass
    for name in ("SC_AVPHYS_PAGES", "SC_PHYS_PAGES"):
        try:
            pages = os.sysconf(name)
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            continue
        if pages > 0 and page_size > 0:
            return pages * page_size, f"sysconf({name})"
    return None, "no host memory API on this platform"


def device_memory_budget(
    device: str | torch.device | None = None,
    *,
    fraction: float = DEFAULT_MEMORY_FRACTION,
    total_bytes: int | None = None,
) -> MemoryBudget:
    """How many bytes a sweep may reach for on ``device``.

    Args:
        device: defaults to :func:`resolve_device`.
        fraction: share of *free* memory the sweep may use.  The remainder is
            headroom for allocator fragmentation and for the rest of the
            notebook.
        total_bytes: pretend the device has this much memory in total.  This is
            how a laptop plans a sweep for the class machine -- pass
            :data:`T4_TOTAL_BYTES` and see which points the T4 will skip.
    """
    torch_device = resolve_device(device)
    name = str(torch_device)

    if total_bytes is not None:
        return MemoryBudget(
            device=name,
            usable_bytes=int(fraction * total_bytes),
            total_bytes=int(total_bytes),
            free_bytes=None,
            fraction=fraction,
            source=f"override: {format_bytes(int(total_bytes))} total x {fraction:g}",
        )

    if torch_device.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(torch_device)
        return MemoryBudget(
            device=name,
            usable_bytes=int(fraction * free),
            total_bytes=int(total),
            free_bytes=int(free),
            fraction=fraction,
            source=f"torch.cuda.mem_get_info: {format_bytes(int(free))} free x {fraction:g}",
        )

    if torch_device.type == "mps":
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        allocated = getattr(torch.mps, "driver_allocated_memory", None)
        if callable(recommended):
            total = int(recommended())
            used = int(allocated()) if callable(allocated) else 0
            free = max(total - used, 0)
            return MemoryBudget(
                device=name,
                usable_bytes=int(fraction * free),
                total_bytes=total,
                free_bytes=free,
                fraction=fraction,
                source=f"torch.mps.recommended_max_memory x {fraction:g}",
            )

    available, source = _host_memory_bytes()
    if available is None:
        warnings.warn(
            f"cannot determine a memory budget for {name} ({source}); the sweep "
            f"will skip nothing and a point that does not fit will raise",
            BenchWarning,
            stacklevel=2,
        )
        return MemoryBudget(
            device=name,
            usable_bytes=None,
            total_bytes=None,
            free_bytes=None,
            fraction=fraction,
            source=source,
        )
    return MemoryBudget(
        device=name,
        usable_bytes=int(fraction * available),
        total_bytes=None,
        free_bytes=int(available),
        fraction=fraction,
        source=f"{source}: {format_bytes(int(available))} x {fraction:g}",
    )


_BYTE_UNITS = (("TiB", 1 << 40), ("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


def format_bytes(nbytes: float | None) -> str:
    """Binary units, exact when the value divides evenly -- as the handout prints."""
    if nbytes is None:
        return "n/a"
    nbytes = int(nbytes)
    if nbytes == 0:
        return "0 B"
    for name, size in _BYTE_UNITS:
        if abs(nbytes) >= size:
            if nbytes % size == 0:
                return f"{nbytes // size} {name}"
            return f"{nbytes / size:.2f} {name}"
    return f"{nbytes:,} B"


# --------------------------------------------------------------------------- #
# spec section 1's two numbers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LinearAttentionLayer:
    """One linear-attention layer, priced by spec section 1.

    Per-layer state bytes at context length ``N``: ``H * d_k * d_v *
    dtype_bytes`` -- **independent of ``N``**.  Marginal bytes per token, per
    layer: **zero**.

    ``d_k`` and ``d_v`` are independent (spec section 2): state bytes go as
    ``d_k * d_v`` while predicted capacity goes as ``d_k``, which is the whole
    point of the homework's state-bytes-matched extension.
    """

    heads: int
    d_k: int
    d_v: int
    dtype_bytes: int = 4

    def state_bytes(self, n: int = 0) -> int:
        """Per-layer state bytes at context length ``n``; the same at every ``n``."""
        return self.heads * self.d_k * self.d_v * self.dtype_bytes

    def marginal_bytes_per_token(self) -> int:
        """Marginal bytes per token, per layer: zero, at every context length."""
        return 0

    @property
    def label(self) -> str:
        return f"linear (H={self.heads}, d_k={self.d_k}, d_v={self.d_v})"


@dataclass(frozen=True)
class SoftmaxAttentionLayer:
    """One softmax-attention (GQA) layer, priced by spec section 1.

    Per-layer state bytes at context length ``N``: ``2 * n_kv_heads * head_dim *
    dtype_bytes * N`` -- **linear in ``N``**.  Marginal bytes per token, per
    layer: ``2 * n_kv_heads * head_dim * dtype_bytes`` -- constant and
    **non-zero**.

    Write ``n_kv_heads``, not ``n_heads``: Llama-3-8B is GQA with 8 KV heads and
    using 32 gives 64 GiB where the answer is 16 GiB.
    """

    n_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    def state_bytes(self, n: int) -> int:
        """Per-layer state bytes at context length ``n``."""
        return 2 * self.n_kv_heads * self.head_dim * self.dtype_bytes * n

    def marginal_bytes_per_token(self) -> int:
        """Marginal bytes per token, per layer -- constant and non-zero."""
        return 2 * self.n_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def label(self) -> str:
        return f"softmax (n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim})"


AttentionLayer = LinearAttentionLayer | SoftmaxAttentionLayer


def state_bytes(layer: AttentionLayer, n: int) -> int:
    """Per-layer state bytes at context length ``n`` (spec section 1)."""
    return layer.state_bytes(n)


def marginal_bytes_per_token(layer: AttentionLayer) -> int:
    """Marginal bytes per token, per layer (spec section 1).

    Note the two names are not interchangeable and the phrase that merges them is
    banned by spec section 1: it is this number for softmax attention and a
    meaningless ``1/N`` quantity for everything else.  Say which one you mean.
    """
    return layer.marginal_bytes_per_token()


# --------------------------------------------------------------------------- #
# run identity (spec I8)
# --------------------------------------------------------------------------- #
Status = Literal["ok", "skipped", "planned"]


@dataclass(frozen=True)
class RunKey:
    """Everything that identifies one benchmarked configuration.

    ``d_k`` and ``d_v`` are **separate fields**, here and in every sweep helper.
    Collapsing them into a single ``d`` makes the homework's state-bytes-matched
    comparison -- ``(d_k=64, d_v=16)`` against ``(d_k=32, d_v=32)`` at equal
    state bytes -- unconstructible (spec I8, spec section 5 Part 3).

    ``warmup`` and ``iters`` are deliberately *not* part of run identity: they
    change the precision of a diagnostic number, not which experiment was run.
    """

    form: str
    rule: str
    b: int
    h: int
    n: int
    d_k: int
    d_v: int
    chunk_size: int | None = DEFAULT_CHUNK_SIZE
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def key_string(self) -> str:
        """A stable identity string -- a resumable sweep keys its results on this."""
        chunk = "none" if self.chunk_size is None else str(self.chunk_size)
        return (
            f"form={self.form}|rule={self.rule}|B={self.b}|H={self.h}|N={self.n}"
            f"|d_k={self.d_k}|d_v={self.d_v}|L={chunk}"
            f"|dtype={self.dtype}|device={self.device}|seed={self.seed}"
        )

    def state_layer(self) -> LinearAttentionLayer:
        """The linear-attention layer this run's state corresponds to."""
        return LinearAttentionLayer(
            heads=self.h, d_k=self.d_k, d_v=self.d_v, dtype_bytes=dtype_bytes(self.dtype)
        )

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.key_string()


# --------------------------------------------------------------------------- #
# predicted footprints
# --------------------------------------------------------------------------- #
def _io_bytes(key: RunKey, n: int, elem: int) -> int:
    """q, k, v and the output: what every form holds before it does anything."""
    return key.b * key.h * n * (2 * key.d_k + 2 * key.d_v) * elem


def _parallel_bytes(key: RunKey) -> int:
    """Two ``(B, H, N, N)`` tensors plus the mask (trap T9).

    ``scores`` is live while ``scores * causal`` is being built, so both exist at
    once; the ``(N, N)`` mask is materialized in the input dtype on top.
    """
    elem = dtype_bytes(key.dtype)
    bh = key.b * key.h
    return (
        _io_bytes(key, key.n, elem)
        + 2 * bh * key.n * key.n * elem
        + key.n * key.n * elem
    )


#: Conservative counts of simultaneously live ``(B, H, C, L, L)`` tensors inside
#: a chunk, per rule.  Gated is the worst: the exponent, the masked copy, the
#: exponentiated decay matrix, and the scores it multiplies.
_CHUNK_SCORE_COPIES: Mapping[str, int] = {"linear": 2, "gated": 4, "delta": 3}


def _chunkwise_bytes(key: RunKey) -> int:
    elem = dtype_bytes(key.dtype)
    bh = key.b * key.h
    length = key.chunk_size or DEFAULT_CHUNK_SIZE
    n_chunks = -(-key.n // length)
    padded = n_chunks * length
    copies = _CHUNK_SCORE_COPIES.get(key.rule, 4)
    return (
        _io_bytes(key, padded, elem)
        + copies * bh * padded * length * elem  # the intra-chunk score blocks
        + 3 * bh * padded * key.d_v * elem  # intra, the output list, the stack
        + bh * n_chunks * key.d_k * key.d_v * elem  # per-chunk state folds
        + length * length  # the boolean mask
    )


def _recurrent_bytes(key: RunKey) -> int:
    elem = dtype_bytes(key.dtype)
    bh = key.b * key.h
    per_step = bh * key.d_k * key.d_k * elem if key.rule == "delta" else 0
    return (
        _io_bytes(key, key.n, elem)
        + 2 * bh * key.n * key.d_v * elem  # the per-token output list, then the stack
        + bh * key.d_k * key.d_v * elem  # the state
        + 4 * per_step  # the erase factor and its friends, one step at a time
    )


def predicted_peak_bytes(key: RunKey) -> int:
    """Upper-ish estimate of the memory one point needs, for the skip decision.

    This is a model of the implementations in :mod:`linattn.reference` and
    :mod:`linattn.chunkwise`, not a measurement: it counts the tensors those
    functions hold at once and prices them in the run's dtype.  It exists so a
    sweep can decide *before* allocating, which is the only place the decision
    can be taken without an OOM (trap T9).  A form registered later supplies its
    own estimator through :class:`BenchForm`.
    """
    form = get_form(key.form)
    return int(form.predict_bytes(key))


# --------------------------------------------------------------------------- #
# the forms under test
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BenchInputs:
    """Deterministic inputs for one run key."""

    key: RunKey
    q: Tensor
    k: Tensor
    v: Tensor
    decay: Tensor | None = None
    beta: Tensor | None = None

    def rule_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"rule": self.key.rule}
        if self.key.rule == "gated":
            kwargs["decay"] = self.decay
        if self.key.rule == "delta":
            kwargs["beta"] = self.beta
        return kwargs


class BenchForm(NamedTuple):
    """One benchmarkable implementation.

    A3's decode path and A5's Triton kernel register themselves here rather than
    forking the sweep engine; see :func:`register_form`.

    Attributes:
        name: the value that appears in :attr:`RunKey.form`.
        run: ``run(inputs) -> Any``, one measured iteration.
        predict_bytes: ``predict_bytes(key) -> int``, the footprint estimate the
            skip decision uses.
        rules: the update rules this form implements.
    """

    name: str
    run: Callable[[BenchInputs], Any]
    predict_bytes: Callable[[RunKey], int]
    rules: tuple[str, ...] = RULES


def _run_parallel(inputs: BenchInputs) -> Tensor:
    return parallel_linear_attention(inputs.q, inputs.k, inputs.v)


def _run_recurrent(inputs: BenchInputs) -> Tensor:
    return recurrent_linear_attention(inputs.q, inputs.k, inputs.v, **inputs.rule_kwargs())


def _run_chunkwise(inputs: BenchInputs) -> Tensor:
    return chunkwise_linear_attention(
        inputs.q,
        inputs.k,
        inputs.v,
        chunk_size=inputs.key.chunk_size or DEFAULT_CHUNK_SIZE,
        **inputs.rule_kwargs(),
    )


#: The registry the sweep dispatches on.  Mutable on purpose -- see
#: :func:`register_form`.
FORMS: dict[str, BenchForm] = {
    "parallel": BenchForm("parallel", _run_parallel, _parallel_bytes, ("linear",)),
    "recurrent": BenchForm("recurrent", _run_recurrent, _recurrent_bytes, RULES),
    "chunkwise": BenchForm("chunkwise", _run_chunkwise, _chunkwise_bytes, RULES),
}


def register_form(form: BenchForm) -> BenchForm:
    """Add (or replace) a benchmarkable form.

    This is the extension point for A3's decode path and A5's Triton kernel: a
    new form arrives with its own footprint estimator and its own set of
    supported rules, and every sweep helper, table, and plot works on it
    unchanged.
    """
    FORMS[form.name] = form
    return form


def get_form(name: str) -> BenchForm:
    try:
        return FORMS[name]
    except KeyError:
        raise ValueError(
            f"unknown form {name!r}; registered forms are {sorted(FORMS)}"
        ) from None


def _validate(key: RunKey) -> None:
    form = get_form(key.form)
    if key.rule not in form.rules:
        raise ValueError(
            f"form {key.form!r} does not implement the {key.rule!r} rule "
            f"(it supports {list(form.rules)}); the {key.form!r} form is the "
            f"linear rule only if that list has one entry"
        )
    if key.n < 0:
        raise ValueError(f"`n` must be >= 0; got {key.n}")
    for name in ("b", "h", "d_k", "d_v"):
        if getattr(key, name) < 1:
            raise ValueError(f"`{name}` must be >= 1; got {getattr(key, name)}")


# --------------------------------------------------------------------------- #
# deterministic inputs
# --------------------------------------------------------------------------- #
def make_bench_inputs(
    key: RunKey,
    *,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> BenchInputs:
    """Deterministic ``q, k, v`` (and ``decay``/``beta``) for one run key.

    Generated **on CPU and then transferred** (spec section 7): the numbers are
    the same on every device, so a curve measured on a laptop and a curve
    measured on the class GPU are curves over the same inputs.

    Delta-rule keys are L2-normalized and ``beta < 1``, because that is the
    regime the real layer occupies (spec I4); raw ``randn`` keys put the
    recurrence somewhere it diverges for reasons unrelated to anything being
    measured.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(key.seed if seed is None else seed)
    dtype = getattr(torch, key.dtype)
    target = torch.device(key.device if device is None else device)

    shape_k = (key.b, key.h, key.n, key.d_k)
    shape_v = (key.b, key.h, key.n, key.d_v)
    q = torch.randn(*shape_k, generator=generator, dtype=dtype)
    k = torch.randn(*shape_k, generator=generator, dtype=dtype)
    v = torch.randn(*shape_v, generator=generator, dtype=dtype)

    decay = beta = None
    if key.rule == "gated":
        # Realistic decays: near 1, so a long chunk does not annihilate the state.
        decay = 0.9 + 0.099 * torch.rand(
            key.b, key.h, key.n, generator=generator, dtype=dtype
        )
    if key.rule == "delta":
        k = k / k.norm(dim=-1, keepdim=True)
        beta = 0.2 + 0.75 * torch.rand(
            key.b, key.h, key.n, generator=generator, dtype=dtype
        )

    def move(t: Tensor | None) -> Tensor | None:
        return None if t is None else t.to(target)

    return BenchInputs(
        key=key, q=move(q), k=move(k), v=move(v), decay=move(decay), beta=move(beta)
    )


def prefill_state(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    rule: Rule = "linear",
    decay: Tensor | None = None,
    beta: Tensor | None = None,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Deprecated alias for :func:`linattn.prefill` (A3).

    A4 and A3 were written concurrently and each grew a prefill.  A3's is the
    canonical one; this name is kept only so benchmark code reads in benchmark
    vocabulary.  It delegates, so there is exactly one implementation and the
    two cannot drift.

    **Trap T11.**  A benchmark that wants a state to decode from must take the
    one the chunkwise pass already holds.  Re-deriving it with a per-token
    Python loop is 32768 interpreter iterations per measured point at a 32k
    context, and it defeats the purpose of the chunkwise form -- which is the
    thing being demonstrated.
    """
    from linattn.decoding import prefill

    return prefill(
        q,
        k,
        v,
        chunk_size=chunk_size,
        rule=rule,
        decay=decay,
        beta=beta,
        initial_state=initial_state,
    )


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SweepPoint:
    """One point of a sweep: measured, planned, or skipped with a reason."""

    key: RunKey
    status: Status
    predicted_bytes: int
    seconds: float | None = None
    timing: TimingResult | None = None
    peak_bytes: int | None = None
    memory: MemoryResult | None = None
    skip_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"

    def row(self) -> dict[str, Any]:
        row = self.key.as_dict()
        row.update(
            status=self.status,
            seconds=self.seconds,
            peak_bytes=self.peak_bytes,
            predicted_bytes=self.predicted_bytes,
            skip_reason=self.skip_reason,
        )
        return row


#: Which :class:`RunKey` field each sweep axis varies.  ``"head_dims"`` varies
#: two at once, which is the point of it.
_AXIS_FIELDS: Mapping[str, tuple[str, ...]] = {
    "n": ("n",),
    "chunk_size": ("chunk_size",),
    "head_dims": ("d_k", "d_v"),
}


@dataclass(frozen=True)
class SweepResult:
    """The points of one sweep, plus everything needed to read them honestly."""

    points: tuple[SweepPoint, ...]
    axis: str
    warmup: int
    iters: int
    dry_run: bool
    budget: MemoryBudget | None = None

    @property
    def ok(self) -> tuple[SweepPoint, ...]:
        return tuple(p for p in self.points if p.status == "ok")

    @property
    def skipped(self) -> tuple[SweepPoint, ...]:
        return tuple(p for p in self.points if p.skipped)

    @property
    def forms(self) -> tuple[str, ...]:
        seen: list[str] = []
        for point in self.points:
            if point.key.form not in seen:
                seen.append(point.key.form)
        return tuple(seen)

    def for_form(self, form: str) -> tuple[SweepPoint, ...]:
        return tuple(p for p in self.points if p.key.form == form)

    def axis_value(self, point: SweepPoint) -> Any:
        fields = _AXIS_FIELDS.get(self.axis, ("n",))
        values = tuple(getattr(point.key, name) for name in fields)
        return values[0] if len(values) == 1 else values

    def axis_label(self, point: SweepPoint) -> str:
        value = self.axis_value(point)
        return "x".join(str(v) for v in value) if isinstance(value, tuple) else str(value)

    def rows(self) -> list[dict[str, Any]]:
        return [p.row() for p in self.points]

    def describe(self) -> str:
        """One line naming every fixed parameter and every swept one.

        ``B`` and ``H`` are always in it: trap T9 requires a sweep over ``N`` to
        state them, and a curve that does not is unreadable.
        """
        if not self.points:
            return "empty sweep"
        labels = {
            "form": "form", "rule": "rule", "b": "B", "h": "H", "n": "N",
            "d_k": "d_k", "d_v": "d_v", "chunk_size": "L", "dtype": "dtype",
            "device": "device", "seed": "seed",
        }
        parts: list[str] = []
        for field_name, label in labels.items():
            values = []
            for point in self.points:
                value = getattr(point.key, field_name)
                if value not in values:
                    values.append(value)
            if len(values) == 1:
                parts.append(f"{label}={values[0]}")
            else:
                parts.append(f"{label}=[{','.join(str(v) for v in values)}]")
        parts.append(f"warmup={self.warmup}, iters={self.iters}")
        if self.dry_run:
            parts.append("DRY RUN (nothing was executed)")
        return ", ".join(parts)

    def format_table(self) -> str:
        """A text table in which a skipped point is a labelled gap, not a hole."""
        lines = [self.describe()]
        if self.budget is not None:
            lines.append(f"memory budget: {self.budget.describe()}")
        header = f"{'form':<12}{self.axis:>10}{'seconds':>14}{'peak':>12}{'predicted':>12}  status"
        lines.append(header)
        lines.append("-" * len(header))
        for point in self.points:
            seconds = "-" if point.seconds is None else f"{point.seconds * 1e3:.3f} ms"
            peak = "-" if point.peak_bytes is None else format_bytes(point.peak_bytes)
            status = "SKIPPED" if point.skipped else point.status
            lines.append(
                f"{point.key.form:<12}{self.axis_label(point):>10}{seconds:>14}"
                f"{peak:>12}{format_bytes(point.predicted_bytes):>12}  {status}"
            )
            if point.skip_reason:
                lines.append(f"{'':<12}{'':>10}  reason: {point.skip_reason}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.format_table()


def _is_out_of_memory(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_device_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and callable(getattr(torch.mps, "empty_cache", None)):
        torch.mps.empty_cache()


def run_sweep(
    keys: Iterable[RunKey],
    *,
    device: str | torch.device | None = None,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    budget_bytes: int | None = None,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    measure_memory: bool = True,
    dry_run: bool = False,
    axis: str = "n",
    reduce: str = "median",
    on_point: Callable[[SweepPoint], None] | None = None,
) -> SweepResult:
    """Run every key, skipping the points that do not fit, and record why.

    Args:
        keys: the configurations to run.  Validated up front, so a bad one fails
            before anything is allocated.
        device: retarget every key onto this device.  ``None`` keeps each key's
            own device.
        warmup, iters: passed to :func:`time_callable`.  Small by default.
        budget_bytes: usable bytes to assume instead of measuring the device.
            Pass :data:`T4_TOTAL_BYTES` to plan a sweep for the class GPU from a
            laptop.
        memory_fraction: share of free memory a measured budget may use.
        measure_memory: take a peak-memory reading (a separate untimed call).
            Silently produces no series on a device with no peak-memory API --
            see :class:`MemoryResult`.
        dry_run: price and validate every point but execute nothing.  Points that
            fit come back with status ``"planned"``.
        axis: which key field this sweep varies -- ``"n"``, ``"chunk_size"``, or
            ``"head_dims"``.  Drives the tables and the x-axis of the plots.
        on_point: called with each :class:`SweepPoint` as it completes, for a
            progress bar in the notebook.

    Returns:
        A :class:`SweepResult`.  Its ``describe()`` states ``B`` and ``H``
        (trap T9) and both head dimensions (spec I8).
    """
    keys = [
        dataclasses.replace(key, device=str(torch.device(device))) if device is not None
        else key
        for key in keys
    ]
    for key in keys:
        _validate(key)

    sweep_device = resolve_device(device if device is not None else (keys[0].device if keys else None))
    if budget_bytes is not None:
        budget = MemoryBudget(
            device=str(sweep_device),
            usable_bytes=int(budget_bytes),
            total_bytes=None,
            free_bytes=None,
            fraction=1.0,
            source=f"explicit budget_bytes={format_bytes(int(budget_bytes))}",
        )
    else:
        budget = device_memory_budget(sweep_device, fraction=memory_fraction)

    points: list[SweepPoint] = []
    for key in keys:
        form = get_form(key.form)
        predicted = int(form.predict_bytes(key))

        if not budget.fits(predicted):
            reason = (
                f"does not fit: predicted {format_bytes(predicted)} > "
                f"budget {format_bytes(budget.usable_bytes)} ({budget.source})"
            )
            point = SweepPoint(
                key=key, status="skipped", predicted_bytes=predicted, skip_reason=reason
            )
            LOGGER.info("SKIP %s -- %s", key.key_string(), reason)
            points.append(point)
            if on_point is not None:
                on_point(point)
            continue

        if dry_run:
            point = SweepPoint(key=key, status="planned", predicted_bytes=predicted)
            LOGGER.info("PLAN %s -- predicted %s", key.key_string(), format_bytes(predicted))
            points.append(point)
            if on_point is not None:
                on_point(point)
            continue

        key_device = torch.device(key.device)
        inputs = None
        try:
            inputs = make_bench_inputs(key)
            memory = None
            if measure_memory:
                _, memory = measure_peak_memory(
                    lambda: form.run(inputs), device=key_device
                )
            timing = time_callable(
                lambda: form.run(inputs),
                device=key_device,
                warmup=warmup,
                iters=iters,
                reduce=reduce,
                label=key.key_string(),
            )
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            if not _is_out_of_memory(exc):
                raise
            reason = (
                f"ran out of memory at run time (predicted {format_bytes(predicted)}, "
                f"{budget.describe()}): {type(exc).__name__}"
            )
            point = SweepPoint(
                key=key, status="skipped", predicted_bytes=predicted, skip_reason=reason
            )
            LOGGER.warning("SKIP %s -- %s", key.key_string(), reason)
            points.append(point)
            if on_point is not None:
                on_point(point)
            continue
        finally:
            del inputs
            _free_device_memory(key_device)

        point = SweepPoint(
            key=key,
            status="ok",
            predicted_bytes=predicted,
            seconds=timing.seconds,
            timing=timing,
            peak_bytes=memory.peak_bytes if memory is not None else None,
            memory=memory,
        )
        LOGGER.info(
            "RUN  %s -- %.3f ms, peak %s (predicted %s)",
            key.key_string(),
            timing.seconds * 1e3,
            format_bytes(point.peak_bytes) if point.peak_bytes is not None else "n/a",
            format_bytes(predicted),
        )
        points.append(point)
        if on_point is not None:
            on_point(point)

    return SweepResult(
        points=tuple(points),
        axis=axis,
        warmup=warmup,
        iters=iters,
        dry_run=dry_run,
        budget=budget,
    )


def _keys_for(
    *,
    forms: Sequence[str],
    lengths: Sequence[int] | None = None,
    chunk_sizes: Sequence[int | None] | None = None,
    head_dims: Sequence[tuple[int, int]] | None = None,
    rule: str,
    b: int,
    h: int,
    n: int | None = None,
    d_k: int | None = None,
    d_v: int | None = None,
    chunk_size: int | None = None,
    dtype: torch.dtype | str,
    device: str | torch.device | None,
    seed: int,
) -> list[RunKey]:
    resolved = str(resolve_device(device))
    dtype_name = _dtype_name(dtype)

    def build(**over: Any) -> list[RunKey]:
        return [
            RunKey(
                form=form,
                rule=rule,
                b=b,
                h=h,
                n=over.get("n", n),
                d_k=over.get("d_k", d_k),
                d_v=over.get("d_v", d_v),
                chunk_size=over.get("chunk_size", chunk_size),
                dtype=dtype_name,
                device=resolved,
                seed=seed,
            )
            for form in forms
        ]

    keys: list[RunKey] = []
    if lengths is not None:
        for length in lengths:
            keys.extend(build(n=length))
    elif chunk_sizes is not None:
        for length in chunk_sizes:
            keys.extend(build(chunk_size=length))
    elif head_dims is not None:
        for pair_k, pair_v in head_dims:
            keys.extend(build(d_k=pair_k, d_v=pair_v))
    return keys


def sweep_context_length(
    *,
    lengths: Sequence[int],
    forms: Sequence[str] = ("parallel", "chunkwise"),
    rule: str = "linear",
    b: int = 1,
    h: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    dtype: torch.dtype | str = torch.float32,
    device: str | torch.device | None = None,
    seed: int = 0,
    **run_kwargs: Any,
) -> SweepResult:
    """Sweep ``N`` at fixed ``B``, ``H``, ``d_k``, ``d_v``.

    The sweep states its ``B`` and ``H`` (they are in every run key and in
    :meth:`SweepResult.describe`) and skips the points it cannot fit -- both
    required by trap T9, which is not hypothetical: two ``(B, H, N, N)`` tensors
    at ``B*H = 8, N = 16384`` are 16 GiB and OOM a T4.

    ``d_k`` and ``d_v`` are separate arguments here and everywhere (spec I8).
    """
    keys = _keys_for(
        forms=forms, lengths=lengths, rule=rule, b=b, h=h, d_k=d_k, d_v=d_v,
        chunk_size=chunk_size, dtype=dtype, device=device, seed=seed,
    )
    return run_sweep(keys, axis="n", **run_kwargs)


def sweep_chunk_size(
    *,
    chunk_sizes: Sequence[int],
    n: int,
    forms: Sequence[str] = ("chunkwise",),
    rule: str = "linear",
    b: int = 1,
    h: int = 8,
    d_k: int = 64,
    d_v: int = 64,
    dtype: torch.dtype | str = torch.float32,
    device: str | torch.device | None = None,
    seed: int = 0,
    **run_kwargs: Any,
) -> SweepResult:
    """Sweep the chunk length ``L`` at fixed everything else -- the U-curve.

    The measured minimum sits above where a pure-FLOP argument puts it, because
    per-chunk overhead in a PyTorch implementation is Python-loop and
    kernel-launch cost, and it moves with the host CPU.  Predict a band and a
    mechanism, not a number (spec section 3, block 4).

    The same sweep is the homework's chunk-size negative control: the chunkwise
    form computes the same function for every ``L`` (spec I2), so accuracy must
    not move even though the wall clock does.
    """
    keys = _keys_for(
        forms=forms, chunk_sizes=list(chunk_sizes), rule=rule, b=b, h=h, n=n,
        d_k=d_k, d_v=d_v, dtype=dtype, device=device, seed=seed,
    )
    return run_sweep(keys, axis="chunk_size", **run_kwargs)


def sweep_head_dims(
    *,
    head_dims: Sequence[tuple[int, int]],
    n: int,
    forms: Sequence[str] = ("chunkwise",),
    rule: str = "linear",
    b: int = 1,
    h: int = 8,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    dtype: torch.dtype | str = torch.float32,
    device: str | torch.device | None = None,
    seed: int = 0,
    **run_kwargs: Any,
) -> SweepResult:
    """Sweep ``(d_k, d_v)`` pairs -- the state-bytes-matched comparison.

    ``head_dims`` is a sequence of ``(d_k, d_v)`` pairs, *not* a sequence of one
    ``d``.  State bytes go as ``d_k * d_v`` while predicted capacity goes as
    ``d_k``, so ``(64, 16)`` against ``(32, 32)`` holds the state size fixed and
    moves the capacity prediction by 2x.  That comparison is the whole point of
    the homework's extension (spec section 5, Part 3) and is unconstructible if
    a scaffold ties the two dimensions together.
    """
    keys = _keys_for(
        forms=forms, head_dims=list(head_dims), rule=rule, b=b, h=h, n=n,
        chunk_size=chunk_size, dtype=dtype, device=device, seed=seed,
    )
    return run_sweep(keys, axis="head_dims", **run_kwargs)

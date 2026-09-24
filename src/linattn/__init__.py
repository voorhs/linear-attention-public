"""linattn -- reference implementations of linear attention for the course unit.

The equivalence ladder (spec I2) starts here: :func:`recurrent_linear_attention`
is ground truth, :func:`parallel_linear_attention` must match it for the linear
rule, :func:`chunkwise_linear_attention` must match it for *every* chunk size
and all three rules, and the Triton kernel added by a later task must match the
chunkwise form in turn.

:func:`chunkwise_linear_attention` is also the training path: it is plain
differentiable PyTorch, and the Triton kernel is forward-only.
"""

from linattn.chunkwise import DEFAULT_CHUNK_SIZE, chunkwise_linear_attention
from linattn.decoding import decode, prefill, step
from linattn.reference import (
    RULES,
    Rule,
    delta_state_update,
    gated_state_update,
    linear_state_update,
    parallel_linear_attention,
    recurrent_linear_attention,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "RULES",
    "Rule",
    "chunkwise_linear_attention",
    "decode",
    "delta_state_update",
    "gated_state_update",
    "linear_state_update",
    "parallel_linear_attention",
    "prefill",
    "recurrent_linear_attention",
    "step",
]

# Benchmarking and figures (plan Task A4).  Appended rather than merged into the
# sorted blocks above so concurrent additions to this file do not collide.
# `linattn.plotting` imports matplotlib lazily, so importing `linattn` still
# works without the optional `plot` extra.
from linattn import bench, plotting  # noqa: E402
from linattn.bench import (  # noqa: E402
    LinearAttentionLayer,
    RunKey,
    SoftmaxAttentionLayer,
    SweepResult,
    measure_peak_memory,
    prefill_state,
    resolve_device,
    run_sweep,
    sweep_chunk_size,
    sweep_context_length,
    sweep_head_dims,
    synchronize,
    time_callable,
)
from linattn.plotting import plot_state_bytes, plot_sweep  # noqa: E402

__all__ += [
    "LinearAttentionLayer",
    "RunKey",
    "SoftmaxAttentionLayer",
    "SweepResult",
    "bench",
    "measure_peak_memory",
    "plot_state_bytes",
    "plot_sweep",
    "plotting",
    "prefill_state",
    "resolve_device",
    "run_sweep",
    "sweep_chunk_size",
    "sweep_context_length",
    "sweep_head_dims",
    "synchronize",
    "time_callable",
]

# The forward-only Triton kernel and the seminar's exercise (plan Task A5).
# Appended rather than merged into the sorted blocks above so concurrent
# additions to this file do not collide.  `linattn.kernel` guards its Triton
# import, so importing `linattn` still works without the optional, Linux-only
# `gpu` extra; `linattn.seminar` is a subpackage and is not imported here.
# The kernel's names are re-exported **lazily**, via PEP 562 module __getattr__.
# `linattn.kernel` guards its Triton import, so an eager re-export would still
# import cleanly without the optional, Linux-only `gpu` extra -- but on a Linux
# box that *has* the extra it would pull Triton into memory for every consumer,
# including `linattn.homework`.  Spec I9 says the homework imports no Triton, so
# this stays lazy: `linattn.triton_chunkwise_linear_attention` resolves on first
# attribute access, and `import linattn.homework` never touches the kernel.
# `linattn.seminar` is a subpackage and is not imported here either.
_KERNEL_EXPORTS = (
    "DEFAULT_KERNEL_CHUNK_SIZE",
    "KernelDomainError",
    "MIN_TL_DOT_DIM",
    "TURING_SHARED_MEMORY_BYTES",
    "TritonUnavailableError",
    "compile_kernel_for_arch",
    "kernel_shared_memory_bytes",
    "launch_chunkwise_kernel",
    "triton_chunkwise_linear_attention",
    "triton_is_available",
)


def __getattr__(name: str):  # PEP 562
    if name in _KERNEL_EXPORTS:
        import linattn.kernel as _kernel

        value = getattr(_kernel, name)
        globals()[name] = value  # resolve once, then it is an ordinary global
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_KERNEL_EXPORTS))


__all__ += list(_KERNEL_EXPORTS)

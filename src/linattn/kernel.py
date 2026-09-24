"""Forward-only chunkwise Triton kernel for the linear rule (plan Task A5).

**Scope.**  This kernel is **forward-only** and implements the **linear rule
only**.  It exists for two things: benchmarking, and teaching tiling.  It is not
a training path and never will be -- **all training everywhere in this unit goes
through** :func:`linattn.chunkwise.chunkwise_linear_attention` **via autograd**,
which is plain differentiable PyTorch and covers all three rules.  There is no
``rule=`` argument here, no backward pass, and no gradient.

The kernel is the last rung of the equivalence ladder (spec I2): A1's recurrent
scan is ground truth, A2's chunkwise form matches it for every chunk size, and
this kernel matches A2.

Notation is spec section 2::

    q, k in R^{d_k}     v in R^{d_v}     S in R^{d_k x d_v}     o_t = S_t^T q_t
    S_t = S_{t-1} + k_t v_t^T                       (the linear rule)

and the three chunk-level identities the kernel implements are A2's, with
``M = tril(ones(L, L))`` the **inclusive-diagonal** mask::

    O = Q S + (M . (Q K^T)) V              S_out = S + K^T V

One program handles one ``(batch, head)`` pair and walks the chunks
sequentially, keeping ``S`` live across the whole sequence.  The grid is
therefore ``B * H`` programs -- small, which is one reason a "faster than
PyTorch" claim is hardware-dependent and is not asserted anywhere (trap T12).

The accepted domain
-------------------
``tl.dot`` requires **every** dimension to be at least 16, and that constrains
``d_k`` and ``d_v`` exactly as much as it constrains the chunk size (trap T12).
The kernel checks this itself and raises :class:`KernelDomainError`; it does not
rely on Triton's own assertion, whose exact form and message are not part of the
documented ``tl.dot`` API and have moved across releases inside the ``>=3.0``
pin.  Accepted:

===================  =======================================================
``chunk_size``       a power of two, ``>= 16``
``d_k``              a power of two, ``>= 16``
``d_v``              a power of two, ``>= 16``; **``d_k != d_v`` is supported**
``N``                any length ``>= 0``; a ragged final chunk is masked
``dtype``            ``torch.float32`` only
device               CUDA
tiling               must fit :data:`TURING_SHARED_MEMORY_BYTES` (see below)
===================  =======================================================

Everything else raises :class:`KernelDomainError` before anything is launched,
so the rejections are testable on a machine with no GPU at all.  A1's asymmetric
pairs ``(d_k=16, d_v=8)`` and ``(d_k=8, d_v=16)`` are out of domain by
construction and are used as rejection fixtures in ``tests/test_kernel.py``;
the kernel's own asymmetric coverage uses ``(32, 16)`` and ``(16, 32)``.

Powers of two are required because the block shape *is* the logical shape: a
chunk size of 65 would compile with ``BLOCK_L = 128`` and quietly cost four
times the shared memory of 64 -- the difference between fitting on the class's
T4 and not.  A2 handles every chunk size, including 1; the kernel does not, and
the tests assert both halves of that.

Shared memory: the budget is Turing's 64 KB, not Ampere's 100 KB
----------------------------------------------------------------
Development happens on an Ampere card with 100 KB of shared memory per SM; the
class runs on a T4 (``sm_75``, Turing) with **64 KB**, and no ``cp.async``.  A
tiling that fits locally may not fit there, so the check is against 64 KB.

Five tiles are live inside the chunk loop -- ``Q``, ``K``, ``V``, the ``L x L``
score block, and the state -- so the requirement is bounded by

    bytes = 4 * (2 L d_k + L d_v + L^2 + d_k d_v)

which is what :func:`kernel_shared_memory_bytes` returns.  Measured by compiling
*this* kernel for both architectures with ``triton.compile`` (see
:func:`compile_kernel_for_arch`; triton 3.8, fp32, ``num_stages=2``):

    L    d_k  d_v   bound     sm_86 actual   sm_75 actual    verdict
    16    16   16    5 120       5 120           3 072       fits both
    16    32   32   11 264      11 264           7 168       fits both
    32    32   32   20 480      20 480          12 288       fits both
    64    32   32   45 056      45 056          32 768       fits both
    32    64   64   45 056      45 056          28 672       fits both
    64    64   64   81 920      81 920          49 152       over the bound
    16   128  128   91 136      91 136          74 752       **fits Ampere,
                                                              not Turing**
   128    32   32  118 784     118 784          98 304       fits neither

Two things this table establishes, neither of them guessed:

* **The bound is exact on ``sm_86`` and conservative on ``sm_75``.**  Turing's
  backend reuses buffers where Ampere's does not, so the analytic sum is an
  upper bound there.  The check uses the bound, because it has to be evaluable
  at call time without invoking a compiler.  The one place inside the tested
  range where that conservatism bites is ``L=64, d_k=d_v=64``: bound 80 KB,
  actual ``sm_75`` requirement 48 KB.  The kernel refuses it.  Widening the
  budget for that tiling is a T4 measurement, not a guess -- see
  :func:`compile_kernel_for_arch`.
* **``L=16, d_k=d_v=128`` is the divergence the spec warns about, measured.**
  Its ``sm_86`` build wants 89 KB and runs on the development card; its
  ``sm_75`` build wants 73 KB and does not fit the T4's 64 KB.  The state tile
  alone is ``d_k * d_v * 4 = 64 KB`` there -- the whole budget.  That is the
  hardware statement of the unit's thesis: the state is the object you are
  trying to keep in SRAM, and its size is ``d_k * d_v``.

Precision: the dot is pinned, and it is not on tensor cores
-----------------------------------------------------------
``tl.dot`` engages TF32 on ``sm_86`` but not on ``sm_75``, so a kernel developed
on Ampere and run on Turing computes different numbers unless the precision is
pinned.  Every dot here passes ``input_precision="ieee"``
(:data:`DOT_INPUT_PRECISION`), which is plain fp32 multiply-add on both
architectures.

Verified by reading the generated PTX for both targets rather than by trusting
the keyword, and against an *unpinned twin* of the same product so the check is
about the pin and not about a no-op.  Measured on triton 3.8:

* unpinned, ``sm_86``: ``mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32``
* unpinned, ``sm_75``: no ``mma.sync`` at all
* pinned, either target: no ``mma.sync`` at all

``tests/test_kernel.py`` asserts all three.  (Do not grep the PTX for the
string ``tf32``: it appears in every build, because Triton writes the source
filename into the debug ``.loc`` lines.  ``mma.sync`` is the instruction.)

**This kernel does not use tensor cores on ``sm_75``.**  An fp32 Triton dot does
not (trap T12), and pinning to ``ieee`` keeps it off the TF32 path on Ampere as
well.  The speedup available here comes from the tiling and from keeping the
state in registers across the whole sequence, not from tensor cores.

Speed: measured and reported, never asserted
--------------------------------------------
No test in this repository asserts that the kernel beats PyTorch.  Against
cuBLAS-backed batched matmuls with a ``B*H`` grid and a register-resident state
that may spill, that comparison is hardware-dependent and must not gate a task
(trap T12).  What was measured, forward only, ``B=H=8``, ``d_k=d_v=32``, fp32,
on the **development** card (RTX 3060 laptop, sm_86) -- ms per forward:

    N       L=16                 L=32                 L=64
            A2     kernel        A2     kernel        A2     kernel
     512    2.19   0.18 (12x)    1.34   0.18 (7.5x)   0.88   0.36 (2.4x)
    2048    8.52   0.52 (16x)    4.76   0.59 (8.0x)   3.19   1.39 (2.3x)
    8192   33.4    2.27 (15x)   18.3    2.44 (7.5x)  12.9    5.44 (2.4x)

Two things worth saying out loud in class.  A2's time falls steeply with ``L``
because its per-chunk cost is a Python iteration and a kernel launch, which is
exactly the mechanism that pushes the U-curve's measured minimum above where a
pure-FLOP argument puts it; the Triton kernel has one launch regardless of
``L``, so its curve is nearly flat and the *ratio* shrinks with ``L`` even
though the kernel gets slightly slower.  And none of these numbers transfer to
the T4 -- they were measured on Ampere, and the kernel's own timing is
sensitive to register spilling, which is architecture-specific.  Re-measure in
the pre-flight session before any of this reaches a slide.

bfloat16 is refused by a check, not by the hardware
---------------------------------------------------
The bf16 ban is scoped to the seminar and this kernel because the reference GPU
is ``sm_75``.  The development card supports bf16 and would run it silently, so
:func:`triton_chunkwise_linear_attention` raises on ``torch.bfloat16`` by
name -- the ban is not left to the hardware to enforce (spec section 6).

What still has to be confirmed on the T4 (charter B4's pre-flight)
------------------------------------------------------------------
Everything above was established on an Ampere card, some of it by compiling for
``sm_75`` without running there.  These five items cannot be closed from here:

1. **The kernel runs at all on ``sm_75``.**  Compiling for a target is not
   executing on it.  Run ``pytest -m gpu`` on the T4.
2. **The numbers match.**  ``tests/test_kernel.py``'s tolerance factor was
   calibrated against residuals measured on ``sm_86`` (14-26x headroom there).
   Re-run the calibration test on the T4; if its residuals are larger, the
   factor moves and the measurement wins.
3. **The shared-memory budget, in the direction that costs us something.**  The
   analytic bound refuses ``L=64, d_k=d_v=64``, whose ``sm_75`` build measures
   48 KB and would fit.  If the seminar wants that tiling, measure it on the T4
   and widen the check deliberately -- do not just raise the constant.
4. **The warp counts.**  ``_launch_config`` was tuned on Ampere, where the
   difference between four and eight warps at ``L=64, d=32`` is 8x.  Register
   files and spill behaviour differ on Turing; re-run the sweep, because a bad
   warp count here does not look like a bug, it looks like the U-curve.
5. **Every timing.**  Nothing in the speed table above transfers.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from linattn.reference import _check_initial_state, _check_sequence_inputs

__all__ = [
    "DEFAULT_KERNEL_CHUNK_SIZE",
    "DOT_INPUT_PRECISION",
    "KERNEL_DTYPE",
    "MIN_TL_DOT_DIM",
    "TURING_ARCH",
    "TURING_SHARED_MEMORY_BYTES",
    "KernelDomainError",
    "TritonUnavailableError",
    "compile_kernel_for_arch",
    "kernel_shared_memory_bytes",
    "launch_chunkwise_kernel",
    "triton_chunkwise_linear_attention",
    "triton_is_available",
]

#: ``tl.dot`` requires every dimension of every operand to be at least this.
#: It constrains ``d_k`` and ``d_v`` as much as it constrains the chunk size.
MIN_TL_DOT_DIM = 16

#: Shared memory per SM on the reference GPU (T4, ``sm_75``, Turing).  The
#: development card has 100 KB; using *that* number is how a tiling passes
#: locally and fails in class.
TURING_SHARED_MEMORY_BYTES = 64 * 1024

#: ``sm_75``.  The architecture the class runs on and the one
#: :func:`compile_kernel_for_arch` defaults to.
TURING_ARCH = 75

#: Half of A2's ``DEFAULT_CHUNK_SIZE``.  The ``L x L`` score tile is the term
#: that grows fastest in the shared-memory budget, and 32 is the largest power
#: of two that leaves room for a 64-wide head on Turing.
DEFAULT_KERNEL_CHUNK_SIZE = 32

#: Pinned so that ``sm_86`` (development) and ``sm_75`` (class) agree.  ``ieee``
#: is plain fp32 multiply-add; it is *not* a tensor-core path on either.
DOT_INPUT_PRECISION = "ieee"

#: fp32 only.  bf16 is banned (``sm_75``); fp16 and fp64 are out of scope.
KERNEL_DTYPE = torch.float32


class KernelDomainError(ValueError):
    """The kernel refuses these inputs, and says which one and why.

    Raised **before any launch**, so a machine with no CUDA device still gets
    real coverage of the domain.  It is a :class:`ValueError` so that callers
    written against A2's validation keep working.
    """


class TritonUnavailableError(RuntimeError):
    """Triton is not importable.  It is an optional, Linux-only dependency."""


try:  # pragma: no cover - the import either works or it does not
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = exc


def triton_is_available() -> bool:
    """``True`` when :mod:`triton` imported.  Every Triton import is guarded and
    a CPU-only ``pytest`` run passes green without it."""
    return triton is not None


# --------------------------------------------------------------------------- #
# domain validation -- all of it before anything is launched
# --------------------------------------------------------------------------- #
def _is_power_of_two(value: int) -> bool:
    return value >= 1 and value & (value - 1) == 0


def kernel_shared_memory_bytes(*, chunk_size: int, d_k: int, d_v: int) -> int:
    """Upper bound on the shared memory one program needs, in bytes.

    Five fp32 tiles are live inside the chunk loop: ``Q`` and ``K``
    (``L x d_k`` each), ``V`` (``L x d_v``), the ``L x L`` score block, and the
    state (``d_k x d_v``).  So::

        bytes = 4 * (2 * L * d_k + L * d_v + L * L + d_k * d_v)

    This is *exactly* what Triton's ``sm_86`` backend allocates for this kernel
    and an over-estimate of what its ``sm_75`` backend allocates -- see the
    measured table in this module's docstring.  It is used as the budget check
    because it can be evaluated at call time without running a compiler.
    """
    floats = 2 * chunk_size * d_k + chunk_size * d_v + chunk_size**2 + d_k * d_v
    return 4 * floats


def _check_dimension(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KernelDomainError(f"`{name}` must be an int; got {value!r}")
    if value < MIN_TL_DOT_DIM:
        raise KernelDomainError(
            f"`{name}` must be at least {MIN_TL_DOT_DIM}: `tl.dot` requires "
            f"every dimension of every operand to be >= {MIN_TL_DOT_DIM}, which "
            f"constrains d_k and d_v as well as the chunk size (spec trap T12). "
            f"Got {name}={value}. Use linattn.chunkwise_linear_attention, which "
            f"has no such restriction."
        )
    if not _is_power_of_two(value):
        raise KernelDomainError(
            f"`{name}` must be a power of two; got {name}={value}. The block "
            f"shape is the logical shape, so a non-power-of-two would be "
            f"rounded up to the next one and quietly cost up to four times the "
            f"shared memory. Use linattn.chunkwise_linear_attention instead."
        )
    return value


def _check_dtype(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16:
        raise KernelDomainError(
            "bfloat16 is banned in this kernel: the reference GPU is a T4 "
            "(sm_75, Turing), which has no bf16 support. The development card "
            "is Ampere and would run it silently, so the ban is enforced here "
            "by a check rather than left to the hardware (spec section 6)."
        )
    if dtype != KERNEL_DTYPE:
        raise KernelDomainError(
            f"the kernel is float32-only; got {dtype}. It is a forward-only "
            f"benchmarking and teaching kernel, and the whole unit runs in "
            f"fp32. Cast, or use linattn.chunkwise_linear_attention."
        )


def _check_shared_memory(*, chunk_size: int, d_k: int, d_v: int) -> None:
    needed = kernel_shared_memory_bytes(chunk_size=chunk_size, d_k=d_k, d_v=d_v)
    if needed > TURING_SHARED_MEMORY_BYTES:
        raise KernelDomainError(
            f"the tiling chunk_size={chunk_size}, d_k={d_k}, d_v={d_v} needs up "
            f"to {needed} bytes of shared memory (Q, K, V, the {chunk_size}x"
            f"{chunk_size} score block and the {d_k}x{d_v} state), over the "
            f"{TURING_SHARED_MEMORY_BYTES}-byte budget of the reference GPU "
            f"(T4, sm_75: 64 KB per SM). The development card has 100 KB, so "
            f"this check is what keeps a tiling that fits locally from failing "
            f"in class. Reduce chunk_size, d_k or d_v."
        )


def _validate_domain(*, chunk_size: object, d_k: int, d_v: int, dtype: torch.dtype) -> int:
    """Everything the kernel refuses, checked in one place before any launch."""
    chunk_size = _check_dimension("chunk_size", chunk_size)
    _check_dimension("d_k", d_k)
    _check_dimension("d_v", d_v)
    _check_dtype(dtype)
    _check_shared_memory(chunk_size=chunk_size, d_k=d_k, d_v=d_v)
    return chunk_size


def _launch_config(chunk_size: int, d_k: int, d_v: int) -> dict[str, int]:
    """``num_warps`` / ``num_stages`` for one tiling.

    ``num_stages=2`` everywhere.  Turing has no ``cp.async``, so the deeper
    pipelines Ampere can build do not transfer, and the chunk loop carries the
    state from one iteration to the next anyway, which limits what there is to
    overlap.  Keeping it fixed means the development card and the T4 run the
    same schedule.

    The warp count is set by the **largest single tile** -- the ``L x L`` score
    block or the ``d_k x d_v`` state, whichever is bigger -- because that is
    what decides register pressure, and register *spilling* is what actually
    costs time here.  Measured on the development card (sm_86, ``B=H=8``,
    ``N=2048``, ms per forward, spilled bytes in brackets):

        L    d     1 warp        2         4          8          16
        16   32   0.84 [136]  0.64 [16]  0.67 [0]   1.21 [0]   3.21 [0]
        32   32   3.60 [592]  1.49 [138] 1.24 [28]  1.76 [0]   2.52 [30]
        64   32  13.83[1518]  7.57 [754] 20.92[1418] 2.62 [96]  8.56 [174]
        16   64  12.54[1226]  6.41 [578] 2.54 [116] 1.88 [12]  7.82 [44]
        32   64  29.04[2874] 18.95[1116] 35.52[1140] 11.01[214] 35.30 [332]

    Four warps win while the largest tile is 32x32; eight win from 64x64 up,
    and sixteen never do.  Getting this wrong is expensive rather than subtly
    slow: ``L=64, d=32`` at four warps spills 1418 bytes and runs **8x** slower
    than the same tiling at eight.
    """
    num_warps = 4 if max(chunk_size**2, d_k * d_v) <= 1024 else 8
    return {"num_warps": num_warps, "num_stages": 2}


# --------------------------------------------------------------------------- #
# the kernel
# --------------------------------------------------------------------------- #
if triton is not None:  # pragma: no branch

    @triton.jit
    def _chunkwise_linear_attention_fwd(
        q_ptr,  # (B, H, N, d_k), contiguous
        k_ptr,  # (B, H, N, d_k), contiguous
        v_ptr,  # (B, H, N, d_v), contiguous
        o_ptr,  # (B, H, N, d_v), contiguous -- written
        state_ptr,  # (B, H, d_k, d_v), contiguous -- read AND written
        N,
        BLOCK_L: tl.constexpr,  # the chunk size L
        BLOCK_DK: tl.constexpr,  # d_k
        BLOCK_DV: tl.constexpr,  # d_v
    ):
        """One program per ``(batch, head)``; the chunks are walked in order.

        Every tensor is contiguous, so the ``(b, h)`` slice of a ``(B,H,N,d)``
        tensor starts at ``pid * N * d`` and its element ``(t, i)`` sits at
        ``t * d + i``.  That is the only pointer arithmetic in the kernel.
        """
        pid = tl.program_id(0)  # = b * H + h

        q_base = q_ptr + pid * N * BLOCK_DK
        k_base = k_ptr + pid * N * BLOCK_DK
        v_base = v_ptr + pid * N * BLOCK_DV
        o_base = o_ptr + pid * N * BLOCK_DV

        offs_l = tl.arange(0, BLOCK_L)
        offs_dk = tl.arange(0, BLOCK_DK)
        offs_dv = tl.arange(0, BLOCK_DV)

        # The state is loaded once and lives in registers for the whole
        # sequence.  This is the point of the chunkwise form: HBM sees each
        # token twice, not once per chunk boundary (trap T11's cousin).
        state_ptrs = (
            state_ptr
            + pid * BLOCK_DK * BLOCK_DV
            + offs_dk[:, None] * BLOCK_DV
            + offs_dv[None, :]
        )
        state = tl.load(state_ptrs)

        # Inclusive of the diagonal: the post-update convention has o_t read a
        # state that already contains token t's own write.
        causal = offs_l[:, None] >= offs_l[None, :]

        for start in range(0, N, BLOCK_L):
            offs_n = start + offs_l
            mask = offs_n < N  # the ragged final chunk

            q = tl.load(
                q_base + offs_n[:, None] * BLOCK_DK + offs_dk[None, :],
                mask=mask[:, None],
                other=0.0,
            )
            k = tl.load(
                k_base + offs_n[:, None] * BLOCK_DK + offs_dk[None, :],
                mask=mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + offs_n[:, None] * BLOCK_DV + offs_dv[None, :],
                mask=mask[:, None],
                other=0.0,
            )

            # The whole chunkwise form, in four lines.  Padded rows carry
            # q = k = v = 0, so they contribute nothing to o and nothing to S.
            scores = tl.where(
                causal, tl.dot(q, tl.trans(k), input_precision="ieee"), 0.0
            )
            o = tl.dot(q, state, input_precision="ieee")
            o += tl.dot(scores, v, input_precision="ieee")
            state += tl.dot(tl.trans(k), v, input_precision="ieee")

            tl.store(
                o_base + offs_n[:, None] * BLOCK_DV + offs_dv[None, :],
                o,
                mask=mask[:, None],
            )

        tl.store(state_ptrs, state)

else:  # pragma: no cover - triton is optional and Linux-only
    _chunkwise_linear_attention_fwd = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# the launcher, shared with the seminar's skeleton and solution
# --------------------------------------------------------------------------- #
def launch_chunkwise_kernel(
    jit_kernel: Any,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    chunk_size: int = DEFAULT_KERNEL_CHUNK_SIZE,
    initial_state: Tensor | None = None,
    return_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Validate, allocate, and launch ``jit_kernel`` over the ``B*H`` grid.

    Factored out so that the seminar's skeleton and solution get **exactly** the
    library's validation, launch configuration and error messages, and differ
    from it only in the kernel body a student fills in.  A student's kernel must
    take the argument list of :func:`_chunkwise_linear_attention_fwd`.

    Args:
        jit_kernel: a ``triton.jit`` function with the argument list above.
        q: ``(B, H, N, d_k)`` float32 CUDA tensor.
        k: ``(B, H, N, d_k)`` float32 CUDA tensor.
        v: ``(B, H, N, d_v)`` float32 CUDA tensor.
        chunk_size: block length ``L``; a power of two ``>= 16``.
        initial_state: optional ``(B, H, d_k, d_v)`` state ``S_0``.  It is
            copied, never written through.
        return_state: when ``True`` also return the final state ``S_N``.

    Returns:
        ``o`` of shape ``(B, H, N, d_v)``, or ``(o, S_N)``.

    Raises:
        KernelDomainError: for anything outside the documented domain, raised
            before any launch and therefore without needing a GPU.
        TritonUnavailableError: when :mod:`triton` is not installed.
    """
    b, h, n, d_k, d_v = _check_sequence_inputs(q, k, v)
    chunk_size = _validate_domain(
        chunk_size=chunk_size, d_k=d_k, d_v=d_v, dtype=q.dtype
    )
    state = _check_initial_state(
        initial_state, b=b, h=h, d_k=d_k, d_v=d_v, dtype=q.dtype, device=q.device
    )

    if q.device.type != "cuda":
        raise KernelDomainError(
            f"the kernel requires CUDA tensors; got device {q.device}. Use "
            f"linattn.chunkwise_linear_attention on CPU -- it computes the same "
            f"function and is the training path everywhere."
        )
    if triton is None:
        raise TritonUnavailableError(
            "triton is not installed; it is an optional, Linux-only extra "
            f"(`pip install 'linattn[gpu]'`). Import error: {_TRITON_IMPORT_ERROR!r}"
        )

    o = torch.empty(b, h, n, d_v, dtype=q.dtype, device=q.device)
    if n == 0:
        return (o, state) if return_state else o

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    state = state.contiguous()
    jit_kernel[(b * h,)](
        q,
        k,
        v,
        o,
        state,
        n,
        BLOCK_L=chunk_size,
        BLOCK_DK=d_k,
        BLOCK_DV=d_v,
        **_launch_config(chunk_size, d_k, d_v),
    )
    return (o, state) if return_state else o


def triton_chunkwise_linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    chunk_size: int = DEFAULT_KERNEL_CHUNK_SIZE,
    initial_state: Tensor | None = None,
    return_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Chunkwise linear attention in Triton -- **forward only, linear rule only**.

    Computes what :func:`linattn.chunkwise.chunkwise_linear_attention` computes
    with ``rule="linear"``, for the subset of shapes named in this module's
    docstring.  It exists for benchmarking and for teaching tiling; **all
    training goes through A2 via autograd**, which is why there is no ``rule=``
    argument and no backward pass here.

    Args:
        q: ``(B, H, N, d_k)`` float32 CUDA tensor; ``d_k`` a power of two >= 16.
        k: ``(B, H, N, d_k)`` float32 CUDA tensor.
        v: ``(B, H, N, d_v)`` float32 CUDA tensor; ``d_v`` a power of two >= 16
            and independent of ``d_k``.
        chunk_size: block length ``L``, a power of two >= 16.  Correctness does
            not depend on it, only speed.
        initial_state: optional ``(B, H, d_k, d_v)`` state ``S_0``, defaulting
            to zeros.  Copied, not written through.
        return_state: when ``True`` also return the final state ``S_N``, which
            the kernel already holds -- no consumer should re-derive it with a
            per-token loop (trap T11).

    Returns:
        ``o`` of shape ``(B, H, N, d_v)``, or ``(o, S_N)`` with ``S_N`` of shape
        ``(B, H, d_k, d_v)`` when ``return_state=True``.

    Raises:
        KernelDomainError: chunk size, ``d_k`` or ``d_v`` below 16 or not a
            power of two; a dtype other than float32 (bfloat16 by name); a
            tiling over Turing's 64 KB of shared memory; a non-CUDA tensor.
            Always raised before any launch.
        TritonUnavailableError: when :mod:`triton` is not installed.

    Example:
        >>> import torch
        >>> from linattn import triton_chunkwise_linear_attention
        >>> q = k = torch.randn(1, 2, 77, 32, device="cuda")   # doctest: +SKIP
        >>> v = torch.randn(1, 2, 77, 16, device="cuda")       # doctest: +SKIP
        >>> o, s = triton_chunkwise_linear_attention(          # doctest: +SKIP
        ...     q, k, v, chunk_size=32, return_state=True)
        >>> o.shape, s.shape                                   # doctest: +SKIP
        (torch.Size([1, 2, 77, 16]), torch.Size([1, 2, 32, 16]))
    """
    return launch_chunkwise_kernel(
        _chunkwise_linear_attention_fwd,
        q,
        k,
        v,
        chunk_size=chunk_size,
        initial_state=initial_state,
        return_state=return_state,
    )


# --------------------------------------------------------------------------- #
# ahead-of-time compilation for the *target* architecture
# --------------------------------------------------------------------------- #
def compile_kernel_for_arch(
    *,
    chunk_size: int,
    d_k: int,
    d_v: int,
    arch: int = TURING_ARCH,
    jit_kernel: Any | None = None,
) -> Any:
    """Compile the kernel for ``arch`` without needing that GPU present.

    This is how the shared-memory budget and the dot-precision pin are checked
    against ``sm_75`` from an Ampere development machine, and it is what the
    pre-flight session (Charter B4) should re-run on the T4 itself.  Returns
    Triton's ``CompiledKernel``; the two fields of interest are
    ``metadata.shared`` (bytes of shared memory) and ``asm["ptx"]``.

    No domain validation happens here on purpose: measuring an over-budget
    tiling is exactly what someone widening the domain needs to do.

    Args:
        chunk_size: block length ``L``.
        d_k: key/query head dimension.
        d_v: value head dimension.
        arch: CUDA compute capability as an int -- ``75`` for the class's T4,
            ``86`` for the development card.
        jit_kernel: a ``triton.jit`` function with this module's argument list;
            defaults to the library kernel.  The seminar's solution passes its
            own so the same measurement covers the student-facing code.

    Raises:
        TritonUnavailableError: when :mod:`triton` is not installed.
    """
    if triton is None:
        raise TritonUnavailableError(
            "triton is not installed; it is an optional, Linux-only extra. "
            f"Import error: {_TRITON_IMPORT_ERROR!r}"
        )
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as triton_compile

    fn = jit_kernel if jit_kernel is not None else _chunkwise_linear_attention_fwd
    constexprs = {"BLOCK_L": chunk_size, "BLOCK_DK": d_k, "BLOCK_DV": d_v}
    signature = {
        name: (
            "*fp32"
            if name.endswith("_ptr")
            else "constexpr"
            if name in constexprs
            else "i32"
        )
        for name in fn.arg_names
    }
    return triton_compile(
        ASTSource(fn=fn, signature=signature, constexprs=constexprs),
        target=GPUTarget("cuda", arch, 32),
        options=_launch_config(chunk_size, d_k, d_v),
    )

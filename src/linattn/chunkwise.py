"""Chunkwise linear attention -- the third view (plan Task A2).

The parallel form is ``O(N^2 d)`` and embarrassingly parallel; the recurrent
form is ``O(N d^2)`` and strictly sequential.  The chunkwise form splits the
sequence into blocks of ``L`` tokens and does both: *inside* a chunk it uses the
parallel form, *between* chunks it uses the recurrence, carrying one
``(B, H, d_k, d_v)`` state across the ``ceil(N / L)`` boundaries.  ``L = 1``
recovers the recurrence and ``L >= N`` recovers the parallel form, so it is one
family with a knob rather than a third algorithm.

This module is the training path for the whole unit: it is written in plain
differentiable PyTorch and everything backpropagates, including through
``decay`` and ``beta``.  The Triton kernel (A5) is forward-only and does not
replace it.

Notation is spec section 2 and matches :mod:`linattn.reference` exactly::

    q, k in R^{d_k}     v in R^{d_v}     S in R^{d_k x d_v}     o_t = S_t^T q_t

    linear:  S_t = S_{t-1} + k_t v_t^T
    gated:   S_t = alpha_t S_{t-1} + k_t v_t^T          (decay before the write)
    delta:   S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T

The state is *post-update*, so the intra-chunk mask is **inclusive of the
diagonal**.  A strict mask paired with folding the chunk's own writes into the
state before reading it is a compensating pair that is exactly right at
``L = 1`` and wrong everywhere else -- which is why the tests sweep four
chunk-size regimes rather than one convenient ``(L, N)``.

The three chunk-level identities
--------------------------------
Write a chunk as ``Q, K in R^{L x d_k}``, ``V in R^{L x d_v}``, incoming state
``S``, and let ``M = tril(ones(L, L))`` be the inclusive-diagonal mask.

*Linear.*  ``S_i = S + sum_{j<=i} k_j v_j^T``, so

    O = Q S + (M * (Q K^T)) V           S_out = S + K^T V

*Gated.*  With ``a_i = sum_{j<=i} log alpha_j`` (inclusive),
``S_i = e^{a_i} S + sum_{j<=i} e^{a_i - a_j} k_j v_j^T``, so

    O = (e^{a} . Q) S + (D * (Q K^T)) V     D_ij = e^{a_i - a_j} for j <= i
    S_out = e^{a_{L-1}} S + (e^{a_{L-1} - a} . K)^T V

Every exponent that survives is ``<= 0``.  The ones that do not survive --
``a_i - a_j`` for ``j > i`` -- are *positive* and grow with ``L``, so ``D`` is
masked **before** the exponential, never after (trap T5; see
:func:`_gated_decay_factors`).

*Delta.*  Rearranging the rule into a rank-1 update,

    S_i = S_{i-1} + k_i w_i^T           w_i = beta_i (v_i - S_{i-1}^T k_i)

and substituting ``S_{i-1} = S + sum_{j<i} k_j w_j^T`` gives a unit
lower-triangular system for the pseudo-values ``W`` (the WY / UT transform)::

    (I + tril(diag(beta) K K^T, -1)) W = diag(beta) (V - K S)

Once ``W`` is known the chunk is *identical to the linear case with*
``V -> W``.  Note that the right-hand side contains ``S``: **the pseudo-values
depend on the incoming state and must be recomputed for every chunk.**  Only
the system matrix is state-independent and can be built for all chunks at once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from linattn.reference import (
    Rule,
    _check_initial_state,
    _check_rule_arguments,
    _check_sequence_inputs,
)

__all__ = ["DEFAULT_CHUNK_SIZE", "chunkwise_linear_attention"]

#: Chunk length used when the caller does not pick one.  Nothing in the maths
#: prefers it; it is a reasonable middle of the U-curve on the reference GPU.
DEFAULT_CHUNK_SIZE = 64


def _check_chunk_size(chunk_size: object) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError(
            f"`chunk_size` must be a positive int; got {chunk_size!r}"
        )
    if chunk_size < 1:
        raise ValueError(f"`chunk_size` must be >= 1; got {chunk_size}")
    return chunk_size


def _to_chunks(x: Tensor, *, pad: int, value: float, n_chunks: int, length: int) -> Tensor:
    """Pad the time axis to ``n_chunks * length`` and split it into chunks.

    ``(B, H, N, d) -> (B, H, C, L, d)`` and ``(B, H, N) -> (B, H, C, L)``.

    The padding is *inert*, not merely ignored: ``k`` and ``v`` are padded with
    zeros (so a padded token writes ``k v^T = 0``), ``decay`` with ones (so it
    neither decays the state nor perturbs the cumulative decay) and ``beta``
    with zeros (so the erase factor is the identity and the write is empty).
    A padded position therefore cannot reach the outputs -- it is causally
    after every real token -- and cannot reach the final state either.
    """
    has_feature_dim = x.ndim == 4
    if pad:
        x = F.pad(x, (0, 0, 0, pad) if has_feature_dim else (0, pad), value=value)
    tail = (length, x.shape[-1]) if has_feature_dim else (length,)
    return x.reshape(x.shape[0], x.shape[1], n_chunks, *tail)


def _gated_decay_factors(
    decay_chunks: Tensor, causal: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The four cumulative-decay quantities a gated chunk needs.

    Args:
        decay_chunks: ``(B, H, C, L)`` per-token ``alpha`` in ``(0, 1]``.
        causal: ``(L, L)`` boolean inclusive-diagonal mask.

    Returns:
        ``(decay_matrix, query_gain, key_gain, state_gain)`` with shapes
        ``(B,H,C,L,L)``, ``(B,H,C,L)``, ``(B,H,C,L)``, ``(B,H,C)``.

    **Trap T5 lives here.**  ``decay_matrix[i, j] = exp(a_i - a_j)`` is wanted
    only for ``j <= i``, where the exponent is non-positive.  Above the diagonal
    the exponent is positive and reaches ``|log alpha| * (L - 1)`` -- at
    ``L = 128, alpha = 0.4`` that is ``e^116``, which overflows fp32.  Zeroing
    those entries *after* the exponential keeps the forward pass finite and
    leaves ``d/dx exp(x) = inf`` sitting in the graph, so the backward pass
    multiplies ``0 * inf`` and returns NaN.  Masking with ``-inf`` **before**
    the exponential makes both passes finite, so the mask is applied first here
    and this ordering is not an optimization to be shuffled.

    ``alpha`` is clamped to the smallest positive normal before the logarithm so
    that a hard zero gives ``exp(-huge) = 0`` rather than ``inf - inf = NaN``.
    """
    tiny = torch.finfo(decay_chunks.dtype).tiny
    log_alpha = decay_chunks.clamp_min(tiny).log()
    cumulative = log_alpha.cumsum(dim=-1)  # a_i, inclusive of token i

    # a_i - a_j, masked to j <= i BEFORE the exponential.
    exponent = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    decay_matrix = exponent.masked_fill(~causal, float("-inf")).exp()

    query_gain = cumulative.exp()  # e^{a_i}: what S_in contributes to o_i
    last = cumulative[..., -1:]
    key_gain = (last - cumulative).exp()  # e^{a_{L-1} - a_j}: decay to chunk end
    state_gain = last.squeeze(-1).exp()  # e^{a_{L-1}}: the carry's own decay
    return decay_matrix, query_gain, key_gain, state_gain


def _delta_system_matrix(key_chunks: Tensor, beta_chunks: Tensor) -> Tensor:
    """``tril(diag(beta) K K^T, -1)`` for every chunk: ``(B, H, C, L, L)``.

    The strictly-lower part is what makes the WY system unit lower triangular
    and hence solvable by forward substitution.  This matrix does *not* depend
    on the incoming state, so it is built once for all chunks; the right-hand
    side does, and is not.

    The caller passes this to ``solve_triangular(..., unitriangular=True)``,
    which never reads the diagonal, so ``tril(-1)`` and ``tril()`` give the same
    answer here.  ``tril(-1)`` is still the right thing to write: it is what the
    derivation says, and it keeps the matrix honest if the ``unitriangular``
    flag is ever dropped (with a zero diagonal the solve would then fail loudly
    instead of quietly using ``beta_i ||k_i||^2`` as a pivot).
    """
    gram = key_chunks @ key_chunks.transpose(-1, -2)  # (k_i . k_j)
    return (beta_chunks.unsqueeze(-1) * gram).tril(-1)


def chunkwise_linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    rule: Rule = "linear",
    decay: Tensor | None = None,
    beta: Tensor | None = None,
    initial_state: Tensor | None = None,
    return_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Chunkwise form of all three rules -- matmuls inside a chunk, the
    recurrence between chunks.

    Computes exactly what :func:`linattn.reference.recurrent_linear_attention`
    computes, for **every** ``chunk_size``, including chunk sizes that do not
    divide ``N`` and chunk sizes larger than ``N``.  The Python loop runs
    ``ceil(N / chunk_size)`` times, not ``N`` times, and everything inside it is
    a batched matmul.

    Args:
        q: ``(B, H, N, d_k)``.
        k: ``(B, H, N, d_k)``.  The delta rule expects L2-normalized keys: with
            ``beta_t ||k_t||^2 > 2`` the recurrence diverges (spec I4).
        v: ``(B, H, N, d_v)``.
        chunk_size: block length ``L >= 1``.  Correctness does not depend on
            it; only speed does.  ``L = 1`` is the recurrence and ``L >= N`` is
            a single parallel block.
        rule: ``"linear"``, ``"gated"``, or ``"delta"``.  Each rule consumes
            exactly one per-timestep argument and supplying the other one is an
            error rather than a silent no-op, exactly as in A1.
        decay: ``(B, H, N)`` decay ``alpha_t`` in ``(0, 1]``, required by and
            only by ``rule="gated"``.  Applied **before** the write, so a
            token's own write is never decayed.
        beta: ``(B, H, N)`` write strength, required by and only by
            ``rule="delta"``.  Production ``beta`` is a sigmoid, hence ``< 1``.
        initial_state: optional ``(B, H, d_k, d_v)`` state ``S_0`` carried in
            from an earlier segment.  Defaults to zeros.  For the delta rule
            this feeds the pseudo-value solve, so it is not merely an additive
            term.
        return_state: when ``True`` also return the final state ``S_N``.

    Returns:
        ``o`` of shape ``(B, H, N, d_v)``, or ``(o, S_N)`` with ``S_N`` of shape
        ``(B, H, d_k, d_v)`` when ``return_state=True``.  ``S_N`` is the state
        the loop already holds -- a prefill must take it from here and never
        re-derive it with a per-token loop (trap T11).

    Raises:
        ValueError: on a non-positive or non-integer ``chunk_size``, an unknown
            rule, a missing or superfluous ``decay`` / ``beta``, or malformed
            shapes, dtypes, or devices.

    Example:
        >>> import torch
        >>> from linattn import chunkwise_linear_attention
        >>> q = k = torch.randn(1, 1, 5, 4)
        >>> v = torch.randn(1, 1, 5, 3)
        >>> o, s = chunkwise_linear_attention(q, k, v, chunk_size=2, return_state=True)
        >>> o.shape, s.shape
        (torch.Size([1, 1, 5, 3]), torch.Size([1, 1, 4, 3]))
    """
    b, h, n, d_k, d_v = _check_sequence_inputs(q, k, v)
    _check_rule_arguments(rule, decay, beta, b=b, h=h, n=n, dtype=q.dtype)
    length = _check_chunk_size(chunk_size)
    state = _check_initial_state(
        initial_state, b=b, h=h, d_k=d_k, d_v=d_v, dtype=q.dtype, device=q.device
    )
    if n == 0:
        empty = torch.zeros(b, h, 0, d_v, dtype=q.dtype, device=q.device)
        return (empty, state) if return_state else empty

    n_chunks = -(-n // length)  # ceil
    pad = n_chunks * length - n
    chunked = dict(pad=pad, n_chunks=n_chunks, length=length)
    queries = _to_chunks(q, value=0.0, **chunked)  # (B, H, C, L, d_k)
    keys = _to_chunks(k, value=0.0, **chunked)
    values = _to_chunks(v, value=0.0, **chunked)  # (B, H, C, L, d_v)

    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()

    # ---- everything that does not depend on the carried state ------------- #
    scores = queries @ keys.transpose(-1, -2)  # (B, H, C, L, L)
    if rule == "gated":
        assert decay is not None  # guaranteed by _check_rule_arguments
        # Padded positions get alpha = 1: inert in the cumulative decay.
        decay_chunks = _to_chunks(decay, value=1.0, **chunked)
        decay_matrix, query_gain, key_gain, state_gain = _gated_decay_factors(
            decay_chunks, causal
        )
        scores = scores * decay_matrix  # decay_matrix already carries the mask
        queries = queries * query_gain.unsqueeze(-1)
        state_keys = keys * key_gain.unsqueeze(-1)
    else:
        scores = scores * causal.to(scores.dtype)
        state_gain = None
        state_keys = keys

    if rule == "delta":
        assert beta is not None
        # Padded positions get beta = 0: erase factor I, empty write.
        beta_chunks = _to_chunks(beta, value=0.0, **chunked)
        system = _delta_system_matrix(keys, beta_chunks)  # (B, H, C, L, L)
    else:
        # V is state-independent for these rules, so the intra-chunk product
        # and the chunk's contribution to the state are batched over chunks.
        intra = scores @ values  # (B, H, C, L, d_v)
        chunk_state = state_keys.transpose(-1, -2) @ values  # (B, H, C, d_k, d_v)

    # ---- the sequential part: one iteration per chunk, not per token ------ #
    outputs: list[Tensor] = []
    for c in range(n_chunks):
        if rule == "delta":
            # W depends on the incoming state, so it is recomputed per chunk;
            # only `system` above could be hoisted out of this loop.
            rhs = beta_chunks[:, :, c].unsqueeze(-1) * (
                values[:, :, c] - keys[:, :, c] @ state
            )
            pseudo_values = torch.linalg.solve_triangular(
                system[:, :, c], rhs, upper=False, unitriangular=True
            )
            chunk_out = scores[:, :, c] @ pseudo_values
            chunk_fold = state_keys[:, :, c].transpose(-1, -2) @ pseudo_values
        else:
            chunk_out = intra[:, :, c]
            chunk_fold = chunk_state[:, :, c]

        # o_i = (S_in^T q_i, decayed) + the intra-chunk part.  The state is read
        # BEFORE this chunk's writes are folded in -- the writes are already
        # accounted for by the inclusive-diagonal mask.
        outputs.append(queries[:, :, c] @ state + chunk_out)

        if state_gain is None:
            state = state + chunk_fold
        else:
            state = state_gain[:, :, c, None, None] * state + chunk_fold

    o = torch.stack(outputs, dim=2).reshape(b, h, n_chunks * length, d_v)
    o = o[:, :, :n]  # drop the padded tail; it never fed a real position
    return (o, state) if return_state else o

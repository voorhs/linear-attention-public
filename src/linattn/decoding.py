"""Prefill and single-token decode -- the deployment path (plan Task A3).

Generation has two regimes and they want different code.  **Prefill** sees the
whole prompt at once and should use the chunkwise form: ``ceil(N / L)``
sequential steps of batched matmuls.  **Decode** sees one token at a time and
has nothing to batch, so it is the recurrence, one rank-1 update per token --
and, crucially, its cost per token does not depend on how many tokens came
before.  That flat line is the payoff the whole unit is built around, so the
two halves live here together with the state that joins them.

The contract::

    o_prompt, S = prefill(q, k, v, chunk_size=L, rule=..., ...)   # (B,H,N,d_v), (B,H,d_k,d_v)
    o_t,       S = step(q_t, k_t, v_t, S, rule=..., ...)          # (B,H,1,d_v), (B,H,d_k,d_v)

``S`` is ``(B, H, d_k, d_v)`` and never grows.  That is the whole point: it is
the model's entire memory of the past, and its size in bytes is
``H * d_k * d_v * dtype_bytes`` *at every context length*, with a marginal cost
of **zero** bytes per token (spec section 1).

**Trap T11 -- the prefill does not re-derive the state.**  It is a one-line
delegation to :func:`linattn.chunkwise.chunkwise_linear_attention` with
``return_state=True``, because that pass already holds ``S_N`` at the end of its
chunk loop.  Re-deriving it with an ``N``-step Python loop would cost 32768
interpreter iterations at a 32k context and defeat the chunkwise form entirely.
There is no ``return_state`` flag here: handing back the state *is* the reason
to call ``prefill`` rather than the chunkwise form directly.

**The delta rule's incoming state is not an additive term.**  For ``linear`` and
``gated`` the recurrence is affine in ``S_{t-1}`` with a state-independent
write, so "run from zero and add ``S_0 q_t`` afterwards" happens to be correct.
For the delta rule it is not: the write is ``beta_t (v_t - S_{t-1}^T k_t)``,
which *reads* the state it is about to overwrite.  A decode that adds the
carried state post hoc agrees with the chunkwise form on two rules out of three
and silently disagrees on delta.

Notation is spec section 2 and matches :mod:`linattn.reference` and
:mod:`linattn.chunkwise` exactly::

    q, k in R^{d_k}     v in R^{d_v}     S in R^{d_k x d_v}     o_t = S_t^T q_t

    linear:  S_t = S_{t-1} + k_t v_t^T
    gated:   S_t = alpha_t S_{t-1} + k_t v_t^T          (decay before the write)
    delta:   S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T

The state is *post-update*: ``o_t`` reads the state after token ``t`` has been
written.  ``linear`` and ``gated`` reuse A1's update functions directly, so
their algebra -- including the decay ordering -- cannot drift from the
reference.  The delta rule uses the rank-1 rearrangement below.

This module is forward-only in spirit but plain differentiable PyTorch in fact;
nothing here is an ``autograd.Function``.  Training still goes through A2.

The module is ``linattn.decoding`` rather than ``linattn.decode`` on purpose:
``decode`` is a *function* exported from :mod:`linattn`, and naming the module
the same thing makes ``linattn.decode`` mean the function in one place and the
module in another.
"""

from __future__ import annotations

import torch
from torch import Tensor

from linattn.chunkwise import DEFAULT_CHUNK_SIZE, chunkwise_linear_attention
from linattn.reference import (
    Rule,
    _check_initial_state,
    _check_rule_arguments,
    _check_sequence_inputs,
    _outer,
    gated_state_update,
    linear_state_update,
)

__all__ = ["prefill", "step", "decode"]


def _read_state(state: Tensor, x: Tensor) -> Tensor:
    """``S^T x``: ``(B,H,d_k,d_v) x (B,H,d_k) -> (B,H,d_v)``."""
    return torch.einsum("bhkv,bhk->bhv", state, x)


def _check_state(state: Tensor | None, **shape) -> Tensor:
    """A1's state validator, reporting under *this* module's parameter name.

    :func:`step` and :func:`decode` call the carried state ``state``, not
    ``initial_state``, and an error message that names a parameter the caller
    cannot see is a documented signature that is not true (spec I10).  The
    checking itself is A1's -- there is no second copy of it here.
    """
    try:
        return _check_initial_state(state, **shape)
    except ValueError as error:
        raise ValueError(str(error).replace("initial_state", "state")) from error


def _step_from_state(
    state: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    *,
    rule: str,
    decay_t: Tensor | None,
    beta_t: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """One token, no validation: ``(o_t, S_t)`` from ``S_{t-1}``.

    The time axis is already gone here -- ``q_t, k_t`` are ``(B, H, d_k)``,
    ``v_t`` is ``(B, H, d_v)``, ``decay_t`` and ``beta_t`` are ``(B, H)``.  Both
    :func:`step` and :func:`decode` call this, which is what keeps the
    single-token entry point and the loop bitwise identical, and it is
    deliberately free of argument checking so that a decode loop validates once
    rather than once per token.

    The delta rule is written as a **rank-1 update**::

        w_t = beta_t (v_t - S_{t-1}^T k_t)          S_t = S_{t-1} + k_t w_t^T

    which is algebraically the left-multiplied erase factor, expanded::

        (I - beta_t k_t k_t^T) S + beta_t k_t v_t^T
            = S - k_t (beta_t S^T k_t)^T + k_t (beta_t v_t)^T
            = S + k_t [beta_t (v_t - S^T k_t)]^T

    It is also exactly A2's WY/UT transform at ``L = 1``, where the unit
    lower-triangular system is the ``1x1`` identity and the pseudo-value is the
    right-hand side unchanged.  Two reasons to write it this way rather than
    calling :func:`linattn.reference.delta_state_update`: it costs
    ``O(d_k d_v)`` per token instead of materializing and applying a
    ``(d_k, d_k)`` erase matrix, and the ``- S^T k_t`` term makes it plain that
    the carried state is *read* by the write and cannot be added on afterwards.
    """
    if rule == "linear":
        state = linear_state_update(state, k_t, v_t)
    elif rule == "gated":
        assert decay_t is not None  # guaranteed by _check_rule_arguments
        state = gated_state_update(state, k_t, v_t, decay_t)
    else:
        assert beta_t is not None
        pseudo_value = beta_t.unsqueeze(-1) * (v_t - _read_state(state, k_t))
        state = state + _outer(k_t, pseudo_value)
    # o_t = S_t^T q_t -- the state is read AFTER the write (post-update).
    return _read_state(state, q_t), state


def _at(value: Tensor | None, t: int) -> Tensor | None:
    """The ``t``-th column of a ``(B, H, N)`` per-timestep argument, or ``None``."""
    return None if value is None else value[:, :, t]


def prefill(
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
    """Run the prompt through the chunkwise form and hand back ``(o, S_N)``.

    This is :func:`linattn.chunkwise.chunkwise_linear_attention` with
    ``return_state=True`` and nothing else -- deliberately.  The chunk loop
    already holds the final state when it finishes, so the prefill's job is to
    *return* it, not to recompute it (trap T11).  A prefill that re-derived the
    state token by token would run ``N`` interpreter iterations instead of
    ``ceil(N / chunk_size)`` -- 32768 of them at a 32k context, per measured
    point -- and would make the chunkwise form pointless.

    Args:
        q: ``(B, H, N, d_k)``.
        k: ``(B, H, N, d_k)``.  The delta rule expects L2-normalized keys
            (spec I4).
        v: ``(B, H, N, d_v)``.
        chunk_size: block length ``L >= 1``.  Correctness does not depend on it,
            only speed; see A2's U-curve.
        rule: ``"linear"``, ``"gated"``, or ``"delta"``.  Each rule consumes
            exactly one per-timestep argument and supplying the other one is an
            error rather than a silent no-op, exactly as in A1 and A2.
        decay: ``(B, H, N)`` decay ``alpha_t`` in ``(0, 1]``, required by and
            only by ``rule="gated"``.  Applied **before** the write.
        beta: ``(B, H, N)`` write strength, required by and only by
            ``rule="delta"``.  Production ``beta`` is a sigmoid, hence ``< 1``.
        initial_state: optional ``(B, H, d_k, d_v)`` state carried in from an
            earlier segment, so a prompt can be prefilled in pieces.  Defaults
            to zeros.  For the delta rule this feeds the pseudo-value solve.

    Returns:
        ``(o, S_N)`` -- **always both**, with ``o`` of shape ``(B, H, N, d_v)``
        and ``S_N`` of shape ``(B, H, d_k, d_v)``.  There is no
        ``return_state`` flag: the state is the reason this function exists, and
        ``S_N`` is what :func:`step` and :func:`decode` consume.

    Raises:
        ValueError: on a non-positive or non-integer ``chunk_size``, an unknown
            rule, a missing or superfluous ``decay`` / ``beta``, or malformed
            shapes, dtypes, or devices.  Validation is A2's -- this function
            adds none of its own, so the two cannot disagree.

    Example:
        >>> import torch
        >>> from linattn import prefill, step
        >>> q = k = torch.randn(1, 2, 13, 4)
        >>> v = torch.randn(1, 2, 13, 3)
        >>> o_prompt, state = prefill(q, k, v, chunk_size=8)
        >>> o_next, state = step(q[:, :, -1:], k[:, :, -1:], v[:, :, -1:], state)
        >>> o_prompt.shape, o_next.shape, state.shape
        (torch.Size([1, 2, 13, 3]), torch.Size([1, 2, 1, 3]), torch.Size([1, 2, 4, 3]))
    """
    return chunkwise_linear_attention(
        q,
        k,
        v,
        chunk_size=chunk_size,
        rule=rule,
        decay=decay,
        beta=beta,
        initial_state=initial_state,
        return_state=True,
    )


def step(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor | None,
    *,
    rule: Rule = "linear",
    decay: Tensor | None = None,
    beta: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """One decode step: one token in, one token and a new state out.

    ``O(d_k d_v)`` work and ``O(d_k d_v)`` memory per call, independent of how
    many tokens have already been decoded -- there is no history to re-read,
    only ``state``.  Carry the returned state into the next call::

        o, state = prefill(q, k, v, chunk_size=64)
        for token in generated:
            o_t, state = step(q_t, k_t, v_t, state)

    The incoming ``state`` is never modified in place, so the same prefilled
    state can seed several independent continuations.

    Args:
        q: ``(B, H, 1, d_k)`` -- the time axis is kept so this is a drop-in for
            a length-1 call to A1 or A2.
        k: ``(B, H, 1, d_k)``.  The delta rule expects L2-normalized keys
            (spec I4).
        v: ``(B, H, 1, d_v)``.
        state: ``(B, H, d_k, d_v)`` state ``S_{t-1}``, typically from
            :func:`prefill` or a previous ``step``.  ``None`` starts from zeros
            -- it has no default, because silently starting a fresh state on
            every call is the one bug this signature can prevent.
        rule: ``"linear"``, ``"gated"``, or ``"delta"``.
        decay: ``(B, H, 1)`` decay ``alpha_t``, required by and only by
            ``rule="gated"``.  Applied **before** the write, so the token's own
            write is undecayed -- the same ordering as A1 and A2, because this
            calls A1's :func:`~linattn.reference.gated_state_update` directly.
        beta: ``(B, H, 1)`` write strength, required by and only by
            ``rule="delta"``.

    Returns:
        ``(o_t, S_t)`` with ``o_t`` of shape ``(B, H, 1, d_v)`` and ``S_t`` of
        shape ``(B, H, d_k, d_v)``.

    Raises:
        ValueError: if the sequence length is not exactly 1, on an unknown rule,
            a missing or superfluous ``decay`` / ``beta``, or malformed shapes,
            dtypes, devices, or state.
    """
    b, h, n, d_k, d_v = _check_sequence_inputs(q, k, v)
    if n != 1:
        raise ValueError(
            f"`step` decodes exactly one token: `q`, `k` and `v` must have "
            f"sequence length 1, got {n}. Use `prefill` for a prompt or "
            f"`decode` for a block of tokens."
        )
    _check_rule_arguments(rule, decay, beta, b=b, h=h, n=n, dtype=q.dtype)
    state = _check_state(
        state, b=b, h=h, d_k=d_k, d_v=d_v, dtype=q.dtype, device=q.device
    )
    o_t, state = _step_from_state(
        state,
        q[:, :, 0],
        k[:, :, 0],
        v[:, :, 0],
        rule=rule,
        decay_t=_at(decay, 0),
        beta_t=_at(beta, 0),
    )
    return o_t.unsqueeze(2), state


def decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor | None,
    *,
    rule: Rule = "linear",
    decay: Tensor | None = None,
    beta: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Decode ``N`` tokens one at a time -- a loop of :func:`step`, validated once.

    Bitwise identical to calling :func:`step` ``N`` times (both go through the
    same unvalidated per-token kernel), but the arguments are checked once
    instead of once per token.  That matters for what this function is *for*:
    the seminar times it against the quadratic form to show decode cost going
    flat in the context length, and per-token Python overhead is exactly what
    would bend that line.

    This is the same recurrence as
    :func:`linattn.reference.recurrent_linear_attention` and must agree with it;
    the difference is that the reference builds the delta rule's ``(d_k, d_k)``
    erase matrix explicitly because that is the definition, while this path uses
    the ``O(d_k d_v)`` rank-1 form because this is the one that ships.

    Args:
        q: ``(B, H, N, d_k)``.
        k: ``(B, H, N, d_k)``.
        v: ``(B, H, N, d_v)``.
        state: ``(B, H, d_k, d_v)`` incoming state, or ``None`` for zeros.
        rule: ``"linear"``, ``"gated"``, or ``"delta"``.
        decay: ``(B, H, N)``, required by and only by ``rule="gated"``.
        beta: ``(B, H, N)``, required by and only by ``rule="delta"``.

    Returns:
        ``(o, S_N)`` with ``o`` of shape ``(B, H, N, d_v)`` and ``S_N`` of shape
        ``(B, H, d_k, d_v)``.

    Raises:
        ValueError: on an unknown rule, a missing or superfluous ``decay`` /
            ``beta``, or malformed shapes, dtypes, devices, or state.
    """
    b, h, n, d_k, d_v = _check_sequence_inputs(q, k, v)
    _check_rule_arguments(rule, decay, beta, b=b, h=h, n=n, dtype=q.dtype)
    state = _check_state(
        state, b=b, h=h, d_k=d_k, d_v=d_v, dtype=q.dtype, device=q.device
    )

    outputs: list[Tensor] = []
    for t in range(n):
        o_t, state = _step_from_state(
            state,
            q[:, :, t],
            k[:, :, t],
            v[:, :, t],
            rule=rule,
            decay_t=_at(decay, t),
            beta_t=_at(beta, t),
        )
        outputs.append(o_t)

    o = (
        torch.stack(outputs, dim=2)
        if outputs
        else torch.zeros(b, h, 0, d_v, dtype=q.dtype, device=q.device)
    )
    return o, state

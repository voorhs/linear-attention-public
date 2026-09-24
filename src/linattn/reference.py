"""Naive reference implementations of linear attention (plan Task A1).

These are the ground truth for the equivalence ladder (spec I2): the chunkwise
form (A2), the decode path (A3), and the Triton kernel (A5) are all checked
against what is in this module.  Clarity beats speed here -- the recurrent scan
is a Python loop over timesteps and the parallel form materializes the full
``(B, H, N, N)`` score matrix on purpose.

Notation (spec section 2, binding everywhere)::

    q, k in R^{d_k}     v in R^{d_v}     S in R^{d_k x d_v}     o_t = S_t^T q_t

* ``q, k`` are ``(B, H, N, d_k)``; ``v`` and ``o`` are ``(B, H, N, d_v)``; the
  state is ``(B, H, d_k, d_v)``.  ``d_k`` and ``d_v`` are independent and are
  never collapsed into a single ``D``.
* **Post-update convention:** ``S_t = sum_{j<=t} k_j v_j^T``, so ``o_t`` reads
  the state *after* token ``t`` has been written.  The causal mask of the
  parallel form is therefore **inclusive of the diagonal**.
* The three update rules, and nothing else (gated+delta combined is out of
  scope for this library; it is lecture material only)::

      linear:  S_t = S_{t-1} + k_t v_t^T
      gated:   S_t = alpha_t S_{t-1} + k_t v_t^T          (decay before the write)
      delta:   S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T

  The delta rule's erase factor multiplies from the **left**.  On the right it
  is not conformable when ``d_k != d_v`` and is wrong even when they are equal
  (spec I1); :func:`delta_state_update` can be asked to build that form purely
  so the tests can demonstrate it fails.
* No ``1/sqrt(d)`` scaling and no ``sum phi(k)`` denominator appear anywhere.
  The lecture motivates linear attention *from* normalized softmax attention,
  so this discrepancy is stated out loud there rather than papered over here.
"""

from __future__ import annotations

from typing import Literal, Mapping

import torch
from torch import Tensor

__all__ = [
    "RULES",
    "Rule",
    "linear_state_update",
    "gated_state_update",
    "delta_state_update",
    "parallel_linear_attention",
    "recurrent_linear_attention",
]

Rule = Literal["linear", "gated", "delta"]

#: The three update rules this library implements.  Test parametrizations in
#: A1/A2/A3 iterate over exactly this tuple.
RULES: tuple[Rule, ...] = ("linear", "gated", "delta")

#: Which per-timestep argument each rule consumes.  Anything else is rejected.
_RULE_ARGUMENT: Mapping[str, str | None] = {
    "linear": None,
    "gated": "decay",
    "delta": "beta",
}


def _outer(a: Tensor, b: Tensor) -> Tensor:
    """``a b^T`` over the trailing dim: ``(..., d_k) x (..., d_v) -> (..., d_k, d_v)``."""
    return a.unsqueeze(-1) * b.unsqueeze(-2)


# --------------------------------------------------------------------------- #
# the three update rules, one line of algebra each
# --------------------------------------------------------------------------- #
def linear_state_update(state: Tensor, k_t: Tensor, v_t: Tensor) -> Tensor:
    """``S_t = S_{t-1} + k_t v_t^T``.

    Args:
        state: ``(B, H, d_k, d_v)`` state ``S_{t-1}``.
        k_t: ``(B, H, d_k)``.
        v_t: ``(B, H, d_v)``.

    Returns:
        ``(B, H, d_k, d_v)`` state ``S_t``.
    """
    return state + _outer(k_t, v_t)


def gated_state_update(
    state: Tensor, k_t: Tensor, v_t: Tensor, decay_t: Tensor
) -> Tensor:
    """``S_t = alpha_t S_{t-1} + k_t v_t^T`` -- decay **before** the write.

    The token's own write is therefore undecayed; in particular ``S_0`` does not
    depend on ``alpha_0`` at all.  Every later implementation (A2's chunkwise
    form, A3's decode) must order it the same way.

    Args:
        state: ``(B, H, d_k, d_v)`` state ``S_{t-1}``.
        k_t: ``(B, H, d_k)``.
        v_t: ``(B, H, d_v)``.
        decay_t: ``(B, H)`` scalar decay ``alpha_t`` per batch and head.

    Returns:
        ``(B, H, d_k, d_v)`` state ``S_t``.
    """
    return decay_t[..., None, None] * state + _outer(k_t, v_t)


def delta_state_update(
    state: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    beta_t: Tensor,
    *,
    erase_side: Literal["left", "right"] = "left",
) -> Tensor:
    """``S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T``.

    One step of online SGD on ``0.5 ||S_{t-1}^T k_t - v_t||^2`` with learning
    rate ``beta_t``.  The erase factor is ``(d_k, d_k)`` and multiplies the
    ``(d_k, d_v)`` state from the **left**.

    Args:
        state: ``(B, H, d_k, d_v)`` state ``S_{t-1}``.
        k_t: ``(B, H, d_k)``.  Keys are expected L2-normalized: with
            ``beta_t ||k_t||^2 > 2`` the erase factor has an eigenvalue below
            ``-1`` and the recurrence diverges (spec I4).
        v_t: ``(B, H, d_v)``.
        beta_t: ``(B, H)`` write strength. Production ``beta`` is a sigmoid, so
            ``beta < 1``.
        erase_side: ``"left"`` is the only correct form and the only one any
            caller should use.  ``"right"`` builds ``S_{t-1}(I - beta k k^T)``,
            which raises :class:`RuntimeError` for non-conformable shapes when
            ``d_k != d_v`` and is silently wrong when they are equal.  It exists
            only so the test-suite can execute spec I1 instead of asserting it
            in a comment.

    Returns:
        ``(B, H, d_k, d_v)`` state ``S_t``.

    Raises:
        ValueError: if ``erase_side`` is neither ``"left"`` nor ``"right"``.
        RuntimeError: from ``torch`` for ``erase_side="right"`` when
            ``d_k != d_v`` -- the shapes do not conform.
    """
    if erase_side not in ("left", "right"):
        raise ValueError(
            f"erase_side must be 'left' (spec section 2) or 'right' (the "
            f"deliberately wrong form, for tests); got {erase_side!r}"
        )
    beta = beta_t[..., None, None]
    d_k = k_t.shape[-1]
    identity = torch.eye(d_k, dtype=state.dtype, device=state.device)
    erase = identity - beta * _outer(k_t, k_t)  # (B, H, d_k, d_k)
    write = beta * _outer(k_t, v_t)  # (B, H, d_k, d_v)
    if erase_side == "left":
        return erase @ state + write
    # (B, H, d_k, d_v) @ (B, H, d_k, d_k): non-conformable unless d_v == d_k.
    return state @ erase + write


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _check_sequence_inputs(q: Tensor, k: Tensor, v: Tensor) -> tuple[int, int, int, int, int]:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, Tensor):
            raise ValueError(f"`{name}` must be a torch.Tensor; got {type(tensor)!r}")
        if tensor.ndim != 4:
            raise ValueError(
                f"`{name}` must be 4-D (B, H, N, d); got shape {tuple(tensor.shape)}"
            )
    if q.shape != k.shape:
        raise ValueError(
            f"`q` and `k` must have the same shape (B, H, N, d_k); got "
            f"{tuple(q.shape)} and {tuple(k.shape)}"
        )
    b, h, n, d_k = q.shape
    if v.shape[:3] != (b, h, n):
        raise ValueError(
            f"`v` must have shape (B, H, N, d_v) with (B, H, N) = {(b, h, n)} "
            f"matching `q`; got shape {tuple(v.shape)}"
        )
    d_v = v.shape[-1]
    for name, tensor in (("k", k), ("v", v)):
        if tensor.dtype != q.dtype:
            raise ValueError(
                f"`{name}` has dtype {tensor.dtype} but `q` has dtype {q.dtype}; "
                f"all inputs must share one dtype"
            )
        if tensor.device != q.device:
            raise ValueError(
                f"`{name}` is on device {tensor.device} but `q` is on {q.device}"
            )
    return b, h, n, d_k, d_v


def _check_initial_state(
    initial_state: Tensor | None,
    *,
    b: int,
    h: int,
    d_k: int,
    d_v: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if initial_state is None:
        return torch.zeros(b, h, d_k, d_v, dtype=dtype, device=device)
    if not isinstance(initial_state, Tensor):
        raise ValueError(
            f"`initial_state` must be a torch.Tensor; got {type(initial_state)!r}"
        )
    if tuple(initial_state.shape) != (b, h, d_k, d_v):
        raise ValueError(
            f"`initial_state` must have shape (B, H, d_k, d_v) = "
            f"{(b, h, d_k, d_v)}; got {tuple(initial_state.shape)}"
        )
    if initial_state.dtype != dtype:
        raise ValueError(
            f"`initial_state` has dtype {initial_state.dtype} but the inputs "
            f"have dtype {dtype}"
        )
    if initial_state.device != device:
        raise ValueError(
            f"`initial_state` is on device {initial_state.device} but the "
            f"inputs are on {device}"
        )
    return initial_state.clone()


def _check_rule_arguments(
    rule: str,
    decay: Tensor | None,
    beta: Tensor | None,
    *,
    b: int,
    h: int,
    n: int,
    dtype: torch.dtype,
) -> None:
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    wanted = _RULE_ARGUMENT[rule]
    for name, value in (("decay", decay), ("beta", beta)):
        if name == wanted:
            if value is None:
                raise ValueError(
                    f"the {rule!r} rule requires `{name}` of shape (B, H, N)"
                )
            if not isinstance(value, Tensor):
                raise ValueError(
                    f"`{name}` must be a torch.Tensor of shape (B, H, N); "
                    f"got {type(value)!r}"
                )
            if tuple(value.shape) != (b, h, n):
                raise ValueError(
                    f"`{name}` must have shape (B, H, N) = {(b, h, n)}; got "
                    f"{tuple(value.shape)}"
                )
            if value.dtype != dtype:
                raise ValueError(
                    f"`{name}` has dtype {value.dtype} but the inputs have "
                    f"dtype {dtype}"
                )
        elif value is not None:
            raise ValueError(
                f"the {rule!r} rule does not use `{name}`; pass `decay` only "
                f"with rule='gated' and `beta` only with rule='delta'"
            )


# --------------------------------------------------------------------------- #
# the two reference forms
# --------------------------------------------------------------------------- #
def parallel_linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    initial_state: Tensor | None = None,
    return_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Quadratic ``O(N^2 d)`` form of the **linear rule only**.

    ``o = tril(q k^T) v`` (plus ``q S_0`` when an incoming state is given).  The
    mask is ``tril`` **including the diagonal**, because the post-update
    convention has ``o_t`` read a state that already contains token ``t``.

    There is no gated or delta variant here: those rules do not regroup into a
    single masked score matrix.  Use :func:`recurrent_linear_attention` for them.

    Args:
        q: ``(B, H, N, d_k)``.
        k: ``(B, H, N, d_k)``.
        v: ``(B, H, N, d_v)``.
        initial_state: optional ``(B, H, d_k, d_v)`` state ``S_0`` carried in
            from an earlier segment.  Defaults to zeros.
        return_state: when ``True`` also return the final state ``S_N``.

    Returns:
        ``o`` of shape ``(B, H, N, d_v)``, or ``(o, S_N)`` with ``S_N`` of shape
        ``(B, H, d_k, d_v)`` when ``return_state=True``.

    Raises:
        ValueError: on malformed shapes, dtypes, or devices.

    Note:
        This form holds two ``(B, H, N, N)`` tensors at once (the scores and the
        masked copy).  At ``B*H = 8, N = 16384`` in fp32 that is 16 GiB -- it
        OOMs a 16 GB card (trap T9).  Any sweep over ``N`` must state its ``B``
        and ``H`` and skip the points it cannot fit.
    """
    b, h, n, d_k, d_v = _check_sequence_inputs(q, k, v)
    state = _check_initial_state(
        initial_state, b=b, h=h, d_k=d_k, d_v=d_v, dtype=q.dtype, device=q.device
    )

    scores = q @ k.transpose(-1, -2)  # (B, H, N, N): scores[t, j] = q_t . k_j
    causal = torch.ones(n, n, dtype=q.dtype, device=q.device).tril()  # j <= t
    o = (scores * causal) @ v
    if initial_state is not None:
        o = o + q @ state  # the S_0 term: (q_t^T S_0)

    if not return_state:
        return o
    final_state = state + k.transpose(-1, -2) @ v
    return o, final_state


def recurrent_linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    rule: Rule = "linear",
    decay: Tensor | None = None,
    beta: Tensor | None = None,
    initial_state: Tensor | None = None,
    return_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Ground-truth recurrent scan: one Python step per token, all three rules.

    At each ``t`` the state is updated first and read second (post-update
    convention)::

        S_t = update(S_{t-1}, k_t, v_t)      o_t = S_t^T q_t

    Args:
        q: ``(B, H, N, d_k)``.
        k: ``(B, H, N, d_k)``.
        v: ``(B, H, N, d_v)``.
        rule: ``"linear"``, ``"gated"``, or ``"delta"`` -- see
            :data:`RULES`.  Each rule consumes exactly one per-timestep
            argument, and supplying the other one is an error rather than a
            silent no-op: ``"gated"`` needs ``decay``, ``"delta"`` needs
            ``beta``, ``"linear"`` needs neither.
        decay: ``(B, H, N)`` scalar decay ``alpha_t`` in ``[0, 1]``, required by
            and only by ``rule="gated"``.  Applied **before** the write.
        beta: ``(B, H, N)`` write strength, required by and only by
            ``rule="delta"``.  Production ``beta`` is a sigmoid, hence ``< 1``.
        initial_state: optional ``(B, H, d_k, d_v)`` state carried in from an
            earlier segment.  Defaults to zeros.
        return_state: when ``True`` also return the final state ``S_N``.

    Returns:
        ``o`` of shape ``(B, H, N, d_v)``, or ``(o, S_N)`` with ``S_N`` of shape
        ``(B, H, d_k, d_v)`` when ``return_state=True``.

    Raises:
        ValueError: on an unknown rule, a missing or superfluous ``decay`` /
            ``beta``, or malformed shapes, dtypes, or devices.
    """
    b, h, n, d_k, d_v = _check_sequence_inputs(q, k, v)
    _check_rule_arguments(rule, decay, beta, b=b, h=h, n=n, dtype=q.dtype)
    state = _check_initial_state(
        initial_state, b=b, h=h, d_k=d_k, d_v=d_v, dtype=q.dtype, device=q.device
    )

    outputs: list[Tensor] = []
    for t in range(n):
        k_t, v_t = k[:, :, t], v[:, :, t]
        if rule == "linear":
            state = linear_state_update(state, k_t, v_t)
        elif rule == "gated":
            assert decay is not None  # guaranteed by _check_rule_arguments
            state = gated_state_update(state, k_t, v_t, decay[:, :, t])
        else:
            assert beta is not None
            state = delta_state_update(state, k_t, v_t, beta[:, :, t])
        # o_t = S_t^T q_t -- the state is read after the write.
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, :, t]))

    o = (
        torch.stack(outputs, dim=2)
        if outputs
        else torch.zeros(b, h, 0, d_v, dtype=q.dtype, device=q.device)
    )
    return (o, state) if return_state else o

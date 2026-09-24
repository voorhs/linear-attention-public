"""Triton-шаг семинара — половина skeleton.

Упражнение — ровно то, что стоит между маркерами ``BEGIN STUDENT BLANK`` и
``END STUDENT BLANK``: четыре оператора, которые *и есть* чанковая форма.  Всё
остальное — арифметика указателей, маски, неполный последний чанк, запуск,
проверки области допустимых входов — дано.

Что ядро вычисляет для одной пары ``(batch, head)``, чанк за чанком, где
``M = tril(ones(L, L))`` — маска **с включённой диагональю** (соглашение «читаем
после обновления»: ``o_t`` читает состояние, уже содержащее токен ``t``)::

    O = Q S + (M . (Q K^T)) V              S_out = S + K^T V

Только прямой проход, только линейное правило.  Всё обучение идёт через
:func:`linattn.chunkwise.chunkwise_linear_attention` посредством autograd.

Две вещи в арифметике, в которых легко ошибиться и которые здесь уже сделаны
правильно:

* каждый ``tl.dot`` фиксирует ``input_precision="ieee"``.  Без фиксации Triton
  выдаёт TF32 ``mma.sync`` на карте разработки (``sm_86``) и обычный fp32 на T4 в
  аудитории (``sm_75``), так что две машины расходились бы численно.
* fp32-произведение **не** выполняется на тензорных ядрах на ``sm_75``.  Ускорение
  берётся из тайлинга и из того, что ``S`` держится в регистрах всю
  последовательность.
"""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor

from linattn.kernel import DEFAULT_KERNEL_CHUNK_SIZE, launch_chunkwise_kernel

__all__ = [
    "skeleton_chunkwise_linear_attention",
    "skeleton_chunkwise_linear_attention_fwd",
]


@triton.jit
def skeleton_chunkwise_linear_attention_fwd(
    q_ptr,  # (B, H, N, d_k), непрерывный
    k_ptr,  # (B, H, N, d_k), непрерывный
    v_ptr,  # (B, H, N, d_v), непрерывный
    o_ptr,  # (B, H, N, d_v), непрерывный — записывается
    state_ptr,  # (B, H, d_k, d_v), непрерывный — читается И записывается
    N,
    BLOCK_L: tl.constexpr,  # размер чанка L
    BLOCK_DK: tl.constexpr,  # d_k
    BLOCK_DV: tl.constexpr,  # d_v
):
    """Одна программа на пару ``(batch, head)``; чанки обходятся по порядку.

    Каждый тензор непрерывен, поэтому срез ``(b, h)`` тензора ``(B,H,N,d)``
    начинается с ``pid * N * d``, а его элемент ``(t, i)`` лежит по смещению
    ``t * d + i``.  Это вся арифметика указателей в ядре.
    """
    pid = tl.program_id(0)  # = b * H + h

    q_base = q_ptr + pid * N * BLOCK_DK
    k_base = k_ptr + pid * N * BLOCK_DK
    v_base = v_ptr + pid * N * BLOCK_DV
    o_base = o_ptr + pid * N * BLOCK_DV

    offs_l = tl.arange(0, BLOCK_L)
    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)

    # Состояние загружается один раз и живёт в регистрах всю последовательность.
    state_ptrs = (
        state_ptr
        + pid * BLOCK_DK * BLOCK_DV
        + offs_dk[:, None] * BLOCK_DV
        + offs_dv[None, :]
    )
    state = tl.load(state_ptrs)

    # Диагональ включена.  Строгая маска — классический неправильный ответ.
    causal = offs_l[:, None] >= offs_l[None, :]

    for start in range(0, N, BLOCK_L):
        offs_n = start + offs_l
        mask = offs_n < N  # неполный последний чанк

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

        # --- BEGIN STUDENT BLANK ---
        # Три вещи, в этом порядке.  Формы, для справки:
        #     q (L, d_k)   k (L, d_k)   v (L, d_v)   state (d_k, d_v)
        #     scores (L, L)             o (L, d_v)
        #
        # TODO 1. `scores`: матрица оценок чанка с маской, включающей диагональ,
        #         M . (Q K^T).  `causal` — маска; tl.where её применяет.
        # TODO 2. `o`: выход этого чанка, Q S + scores V.  S — состояние,
        #         перенесённое из предыдущих чанков; его нужно прочитать *до*
        #         того, как в него попадут записи самого этого чанка, — маска
        #         выше их уже учитывает.
        # TODO 3. внести записи этого чанка в состояние: S += K^T V.
        #
        # Транспонируйте через tl.trans(x) и передавайте input_precision="ieee"
        # в каждое произведение, иначе железо разработки и железо в аудитории
        # разойдутся.
        #
        # Замените три заглушки ниже.  В исходном виде они компилируются и
        # работают, а проверка в следующей ячейке падает — так вы и узнаёте,
        # что ещё не закончили.
        scores = tl.zeros([BLOCK_L, BLOCK_L], dtype=tl.float32)  # TODO 1
        o = tl.zeros([BLOCK_L, BLOCK_DV], dtype=tl.float32)  # TODO 2
        state = state + 0.0  # TODO 3
        # --- END STUDENT BLANK ---

        tl.store(
            o_base + offs_n[:, None] * BLOCK_DV + offs_dv[None, :],
            o,
            mask=mask[:, None],
        )

    tl.store(state_ptrs, state)


def skeleton_chunkwise_linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    chunk_size: int = DEFAULT_KERNEL_CHUNK_SIZE,
    initial_state: Tensor | None = None,
    return_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Запустить ядро выше.

    Валидация, выделение памяти и конфигурация запуска — библиотечные, поэтому
    эта функция отказывает ровно в том, в чём отказывает
    :func:`linattn.kernel.triton_chunkwise_linear_attention`, и с теми же
    сообщениями: размер чанка, ``d_k`` или ``d_v`` меньше 16 или не степень
    двойки, dtype, отличный от float32, и тайлинг, превышающий 64 KB разделяемой
    памяти T4.
    """
    return launch_chunkwise_kernel(
        skeleton_chunkwise_linear_attention_fwd,
        q,
        k,
        v,
        chunk_size=chunk_size,
        initial_state=initial_state,
        return_state=return_state,
    )

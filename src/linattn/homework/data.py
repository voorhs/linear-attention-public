"""MQAR: многозапросное ассоциативное воспроизведение.

Последовательность несёт ``M`` пар ключ→значение.  ``Q`` из этих ключей
повторяются в конце, и модель должна выдать значение, с которым каждый был
связан.  Две независимые ручки-заполнителя развязывают длину последовательности
и ``M`` (ловушка T3 спецификации курса):

``n_pad`` (P)
    Копии одного инертного токена.  Один id, один эмбеддинг, так что миксер
    может научиться ничего для него не записывать.  Вставляются на случайных
    *границах пар*, никогда между ключом и его значением.  Теория предсказывает,
    что паддинг вообще не может сдвинуть кривую ёмкости — это отрицательный
    контроль.

``n_distractors`` (D)
    Дистракторные **токены**, выдаваемые как ``D/2`` соседних пар ключ/значение,
    ключи которых берутся из того же пула, что и настоящие ключи, не пересекаются
    с ними и никогда не запрашиваются.  На уровне токенов они неотличимы от
    настоящих пар.  Теория предсказывает, что каждый стоит половину сохранённой
    пары — это тест на число записей (ловушка T6).

Одна ручка сама по себе делает неконструируемым либо отрицательный контроль,
либо тест на число записей, поэтому обе существуют с первого коммита.

Раскладка::

    [ M настоящих пар и D/2 пар-дистракторов, перемешаны, P паддингов на границах пар ]  [ q_1 .. q_Q ]
    |<--------------------------- контекст, 2M + D + P токенов ------------------------->|    |<-- Q -->|

Цели **выровнены по позиции, а не сдвинуты на следующий токен**: цель на позиции
``q_j`` — это значение, связанное с ним, и
:data:`~linattn.homework.config.IGNORE_INDEX` везде в остальных местах.  Ответы
никогда не подаются обратно в последовательность, и именно поэтому число записей
``W = 2M + D + Q`` — с ``Q``, а не ``2Q`` — и данные согласованы с выводом,
который оценивается в задании.

Всё здесь работает на CPU под явным ``torch.Generator`` (раздел 7 спецификации
курса: генерация данных не зависит от устройства по построению).
"""

from __future__ import annotations

from typing import Tuple

import torch

from .config import IGNORE_INDEX, PAD_ID, RunConfig, derive_seed


def write_count(cfg: RunConfig) -> int:
    """``W = 2M + D + Q`` — записи, стоящие в состоянии на позиции запроса.

    Не ``M``.  Слой записывает внешнее произведение на *каждом* токене, так что
    вывод, считающий сохранённые пары, занижает примерно в 2 раза (ловушка T6).
    Это единственное определение; текст задания и ответы к части 1 цитируют его
    отсюда.
    """
    return 2 * cfg.n_pairs + cfg.n_distractors + cfg.n_queries


def pad_for_seq_len(cfg: RunConfig, seq_len: int) -> int:
    """Такой ``n_pad``, при котором ``cfg`` имеет ровно ``seq_len`` токенов.

    Используется, чтобы держать длину последовательности фиксированной, пока
    меняется ``M``: тогда на оси ёмкости одна независимая переменная
    (инвариант I7), и ``M`` не коллинеарно с ``N`` (ловушка T3).
    """
    need = seq_len - (2 * cfg.n_pairs + cfg.n_distractors + cfg.n_queries)
    if need < 0:
        raise ValueError(
            f"нельзя дополнить до seq_len={seq_len}: 2M+D+Q уже равно "
            f"{2 * cfg.n_pairs + cfg.n_distractors + cfg.n_queries}"
        )
    return need


def _topk_without_replacement(
    n: int, pool: int, k: int, generator: torch.Generator
) -> torch.Tensor:
    """``k`` различных индексов в ``[0, pool)`` на строку, равномерно случайно."""
    noise = torch.rand(n, pool, generator=generator)
    return noise.argsort(dim=1)[:, :k]


def make_dataset(
    cfg: RunConfig,
    n_examples: int,
    stream: str,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Сгенерировать ``n_examples`` примеров MQAR на CPU.

    Параметры
    ---------
    cfg:
        Конфигурация запуска.  ``M``, ``D``, ``P``, ``Q`` и (фиксированный)
        словарь берутся отсюда.
    n_examples:
        Число последовательностей.
    stream:
        Имя потока seed, например ``"train"`` или ``"eval"``.  Два потока
        одного запуска статистически независимы, так что отложенная выборка
        действительно отложена.
    seed:
        Переопределяет ``cfg.seed`` при выводе seed потока.  Тесты этим
        пользуются; запуски — нет.

    Возвращает
    ----------
    tokens, targets:
        Оба ``(n_examples, cfg.seq_len)``, dtype ``int64``, на CPU.
        ``targets`` равно :data:`~linattn.homework.config.IGNORE_INDEX` везде,
        кроме ``Q`` позиций запросов.
    """
    if n_examples < 1:
        raise ValueError("n_examples должно быть >= 1")

    gen = torch.Generator()
    gen.manual_seed(derive_seed(cfg.seed if seed is None else seed, stream))

    n = n_examples
    m = cfg.n_pairs
    n_blocks = cfg.n_pair_blocks  # M настоящих + D/2 пар-дистракторов
    n_pad = cfg.n_pad
    n_slots = n_blocks + n_pad
    context_len = 2 * n_blocks + n_pad
    seq_len = cfg.seq_len

    # --- различные id ключей и значений ----------------------------------
    # Блоки 0..M-1 — настоящие пары; M..n_blocks-1 — дистракторы.  Ключи
    # различны по всему контексту, так что ключ-дистрактор никогда не совпадёт
    # с запрашиваемым ключом и задача никогда не двусмысленна.  Значения тоже
    # различны, так что «выдать какое-нибудь значение из контекста» набирает
    # ровно 1/n_blocks, и вырожденное решение опознаётся по одному лишь числу
    # точности.
    keys = _topk_without_replacement(n, cfg.n_key_tokens, n_blocks, gen) + cfg.key_lo
    values = _topk_without_replacement(n, cfg.n_value_tokens, n_blocks, gen) + cfg.value_lo

    # --- перемешать порядок пар -------------------------------------------
    order = torch.rand(n, n_blocks, generator=gen).argsort(dim=1)
    seq_keys = keys.gather(1, order)
    seq_vals = values.gather(1, order)

    # --- расставить P паддингов на границах пар ----------------------------
    # n_slots слотов вмещают n_blocks пар (по 2 токена) и P паддингов (по 1
    # токену).  Случайная перестановка решает, какие слоты — паддинги;
    # кумулятивная сумма длин слотов даёт начальное смещение каждого слота.
    # Поэтому паддинги попадают между парами и никогда не разрывают пару.
    tokens = torch.full((n, seq_len), PAD_ID, dtype=torch.long)
    if n_pad:
        slot_perm = torch.rand(n, n_slots, generator=gen).argsort(dim=1)
        is_pad_slot = slot_perm < n_pad
    else:
        is_pad_slot = torch.zeros(n, n_slots, dtype=torch.bool)
    slot_len = torch.where(is_pad_slot, 1, 2)
    starts = slot_len.cumsum(dim=1) - slot_len
    # Булево маскирование сохраняет построчный порядок, так что это начальное
    # смещение r-й пары в порядке пар.
    pair_starts = starts[~is_pad_slot].view(n, n_blocks)
    tokens.scatter_(1, pair_starts, seq_keys)
    tokens.scatter_(1, pair_starts + 1, seq_vals)

    # --- запросы -----------------------------------------------------------
    qsel = _topk_without_replacement(n, m, cfg.n_queries, gen)
    query_keys = keys.gather(1, qsel)  # индексируется по id блока, то есть только настоящие пары
    query_vals = values.gather(1, qsel)
    tokens[:, context_len:] = query_keys

    targets = torch.full((n, seq_len), IGNORE_INDEX, dtype=torch.long)
    targets[:, context_len:] = query_vals

    return tokens, targets

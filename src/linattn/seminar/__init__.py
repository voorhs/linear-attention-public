"""The seminar's Triton exercise: the student skeleton and the hidden solution.

**Why this lives in the library rather than in the notebook.**  Charter B2 says
the Triton step is fill-in-the-blank, never from scratch, that the blank stays
small, and that a hidden solution cell ships so a student blocked on syntax can
still finish the seminar.  Anything that only exists inside a notebook is
untested, and "nothing may fail live" is the seminar's hard constraint.  So both
halves of the exercise are ordinary Python modules here, under test in
``tests/test_kernel.py``, and the notebook *displays* them:

    from linattn.seminar import skeleton_source, solution_source
    print(skeleton_source())          # -> the student cell
    print(solution_source())          # -> the hidden cell

:mod:`linattn.seminar.triton_skeleton` and :mod:`linattn.seminar.triton_solution`
are the **same file** outside the two markers below, and a test asserts it.  The
exercise is exactly the lines between the markers -- four statements that are
the chunkwise form itself -- and everything around them (pointer arithmetic,
masks, the ragged tail, the launch, the domain checks) is given.  Both call
:func:`linattn.kernel.launch_chunkwise_kernel`, so a student's kernel gets the
library's validation and error messages for free, and the filled-in skeleton is
the library's kernel rather than a lookalike.

Importing this package does **not** import Triton; importing either of the two
modules does.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "BLANK_BEGIN",
    "BLANK_END",
    "SKELETON_PATH",
    "SOLUTION_PATH",
    "blank_region",
    "skeleton_source",
    "solution_source",
    "split_on_blank",
]

#: The markers that delimit what a student writes.  They are load-bearing: a
#: test asserts the two files are identical outside them.
BLANK_BEGIN = "# --- BEGIN STUDENT BLANK ---"
BLANK_END = "# --- END STUDENT BLANK ---"

SKELETON_PATH = Path(__file__).with_name("triton_skeleton.py")
SOLUTION_PATH = Path(__file__).with_name("triton_solution.py")


def skeleton_source() -> str:
    """The student-facing file, verbatim.  Paste it into the exercise cell."""
    return SKELETON_PATH.read_text(encoding="utf-8")


def solution_source() -> str:
    """The completed file, verbatim.  This is the hidden cell's contents."""
    return SOLUTION_PATH.read_text(encoding="utf-8")


def split_on_blank(source: str) -> tuple[str, str, str]:
    """``(before, blank, after)`` around the student markers.

    Raises:
        ValueError: when a marker is missing, which would mean the exercise had
            lost its blank.
    """
    if BLANK_BEGIN not in source or BLANK_END not in source:
        raise ValueError("the source has no student blank")
    before, rest = source.split(BLANK_BEGIN, 1)
    body, after = rest.split(BLANK_END, 1)
    return before, body, after


def blank_region(source: str) -> str:
    """Just the part a student writes."""
    return split_on_blank(source)[1]

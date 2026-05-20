import pytest

from rubik.rubik_state import SOLVED_STATE, build_state_string, validate_state_string


def test_build_state_string_for_solved_cube() -> None:
    labels_by_face = {
        "U": list("UUUUUUUUU"),
        "R": list("RRRRRRRRR"),
        "F": list("FFFFFFFFF"),
        "D": list("DDDDDDDDD"),
        "L": list("LLLLLLLLL"),
        "B": list("BBBBBBBBB"),
    }
    assert build_state_string(labels_by_face) == SOLVED_STATE


def test_validate_state_string_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        validate_state_string("U")


def test_validate_state_string_rejects_wrong_counts() -> None:
    with pytest.raises(ValueError):
        validate_state_string("U" * 54)

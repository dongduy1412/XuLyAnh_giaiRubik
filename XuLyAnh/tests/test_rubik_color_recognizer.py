import numpy as np
import pytest

from src.rubik.color_recognizer import hsv_distance, validate_color_counts


def test_hsv_distance_wraps_hue() -> None:
    color_a = np.array([1, 100, 100], dtype=np.float32)
    color_b = np.array([179, 100, 100], dtype=np.float32)
    assert hsv_distance(color_a, color_b) == pytest.approx((2.0 * 2 * 2) ** 0.5)


def test_validate_color_counts_accepts_nine_of_each_symbol() -> None:
    labels_by_face = {
        "U": list("UUUUUUUUU"),
        "R": list("RRRRRRRRR"),
        "F": list("FFFFFFFFF"),
        "D": list("DDDDDDDDD"),
        "L": list("LLLLLLLLL"),
        "B": list("BBBBBBBBB"),
    }
    validate_color_counts(labels_by_face)


def test_validate_color_counts_rejects_bad_counts() -> None:
    labels_by_face = {
        "U": list("UUUUUUUUR"),
        "R": list("RRRRRRRRR"),
        "F": list("FFFFFFFFF"),
        "D": list("DDDDDDDDD"),
        "L": list("LLLLLLLLL"),
        "B": list("BBBBBBBBB"),
    }
    with pytest.raises(ValueError):
        validate_color_counts(labels_by_face)

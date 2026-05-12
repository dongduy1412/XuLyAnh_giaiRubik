from collections import Counter
from math import sqrt

import cv2
import numpy as np

from .face_detector import inner_cell_crop
from .rubik_state import FACE_ORDER


DEFAULT_WEIGHTS = (2.0, 1.0, 0.5)


def get_mean_hsv(cell: np.ndarray, margin_ratio: float = 0.25) -> np.ndarray:
    sample = inner_cell_crop(cell, margin_ratio)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    return pixels.mean(axis=0)


def hsv_distance(color_a: np.ndarray, color_b: np.ndarray, weights: tuple[float, float, float] = DEFAULT_WEIGHTS) -> float:
    h1, s1, v1 = color_a.astype(float)
    h2, s2, v2 = color_b.astype(float)
    delta_h = abs(h1 - h2)
    delta_h = min(delta_h, 180 - delta_h)
    delta_s = abs(s1 - s2)
    delta_v = abs(v1 - v2)
    w_h, w_s, w_v = weights
    return sqrt(w_h * delta_h * delta_h + w_s * delta_s * delta_s + w_v * delta_v * delta_v)


def extract_reference_colors(cells_by_face: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    references = {}
    for face in FACE_ORDER:
        cells = cells_by_face.get(face)
        if cells is None or len(cells) != 9:
            raise ValueError(f"Face {face} must contain exactly 9 cells")
        references[face] = get_mean_hsv(cells[4])
    return references


def classify_cell(cell_hsv: np.ndarray, references: dict[str, np.ndarray]) -> tuple[str, float, float]:
    distances = sorted((hsv_distance(cell_hsv, reference), face) for face, reference in references.items())
    best_distance, best_face = distances[0]
    second_distance = distances[1][0] if len(distances) > 1 else best_distance
    confidence_gap = second_distance - best_distance
    return best_face, best_distance, confidence_gap


def recognize_faces(cells_by_face: dict[str, list[np.ndarray]]) -> tuple[dict[str, list[str]], dict[str, list[dict[str, float | str]]]]:
    references = extract_reference_colors(cells_by_face)
    labels_by_face: dict[str, list[str]] = {}
    details_by_face: dict[str, list[dict[str, float | str]]] = {}

    for face in FACE_ORDER:
        labels = []
        details = []
        for index, cell in enumerate(cells_by_face[face]):
            cell_hsv = get_mean_hsv(cell)
            label, distance, confidence_gap = classify_cell(cell_hsv, references)
            labels.append(label)
            details.append({
                "index": index,
                "label": label,
                "distance": round(distance, 3),
                "confidence_gap": round(confidence_gap, 3),
            })
        labels_by_face[face] = labels
        details_by_face[face] = details

    return labels_by_face, details_by_face


def validate_reference_separation(references: dict[str, np.ndarray], min_distance: float = 25.0) -> list[str]:
    warnings = []
    faces = list(references)
    for i, face_a in enumerate(faces):
        for face_b in faces[i + 1:]:
            distance = hsv_distance(references[face_a], references[face_b])
            if distance < min_distance:
                warnings.append(f"Reference colors {face_a} and {face_b} are close: {distance:.2f}")
    return warnings


def color_counts(labels_by_face: dict[str, list[str]]) -> Counter:
    counter = Counter()
    for labels in labels_by_face.values():
        counter.update(labels)
    return counter


def validate_color_counts(labels_by_face: dict[str, list[str]]) -> None:
    counts = color_counts(labels_by_face)
    invalid = {face: counts.get(face, 0) for face in FACE_ORDER if counts.get(face, 0) != 9}
    if invalid:
        raise ValueError(f"Invalid color counts: {invalid}. Each face symbol must appear exactly 9 times.")

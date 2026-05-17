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


def get_reference_distances(cell_hsv: np.ndarray, references: dict[str, np.ndarray]) -> dict[str, float]:
    return {face: hsv_distance(cell_hsv, reference) for face, reference in references.items()}


def classify_cell(cell_hsv: np.ndarray, references: dict[str, np.ndarray]) -> tuple[str, float, float, dict[str, float]]:
    distance_by_face = get_reference_distances(cell_hsv, references)
    distances = sorted((distance, face) for face, distance in distance_by_face.items())
    best_distance, best_face = distances[0]
    second_distance = distances[1][0] if len(distances) > 1 else best_distance
    confidence_gap = second_distance - best_distance
    return best_face, best_distance, confidence_gap, distance_by_face


def rebalance_color_counts(
    labels_by_face: dict[str, list[str]],
    details_by_face: dict[str, list[dict]],
) -> None:
    counts = color_counts(labels_by_face)
    while True:
        over_faces = [face for face in FACE_ORDER if counts.get(face, 0) > 9]
        under_faces = [face for face in FACE_ORDER if counts.get(face, 0) < 9]
        if not over_faces or not under_faces:
            return

        candidates = []
        for source_face in over_faces:
            for image_face in FACE_ORDER:
                for index, label in enumerate(labels_by_face[image_face]):
                    if label != source_face or index == 4:
                        continue
                    distances = details_by_face[image_face][index]["distances"]
                    for target_face in under_faces:
                        penalty = distances[target_face] - distances[source_face]
                        candidates.append((penalty, image_face, index, source_face, target_face))

        if not candidates:
            return

        _, image_face, index, source_face, target_face = min(candidates, key=lambda item: item[0])
        labels_by_face[image_face][index] = target_face
        detail = details_by_face[image_face][index]
        detail["label"] = target_face
        detail["rebalanced_from"] = source_face
        detail["distance"] = round(detail["distances"][target_face], 3)
        counts[source_face] -= 1
        counts[target_face] += 1


def recognize_faces(cells_by_face: dict[str, list[np.ndarray]]) -> tuple[dict[str, list[str]], dict[str, list[dict[str, float | str]]]]:
    references = extract_reference_colors(cells_by_face)
    labels_by_face: dict[str, list[str]] = {}
    details_by_face: dict[str, list[dict[str, float | str]]] = {}

    for face in FACE_ORDER:
        labels = []
        details = []
        for index, cell in enumerate(cells_by_face[face]):
            cell_hsv = get_mean_hsv(cell)
            label, distance, confidence_gap, distance_by_face = classify_cell(cell_hsv, references)
            labels.append(label)
            details.append({
                "index": index,
                "label": label,
                "distance": round(distance, 3),
                "confidence_gap": round(confidence_gap, 3),
                "distances": {key: round(value, 3) for key, value in distance_by_face.items()},
            })
        labels_by_face[face] = labels
        details_by_face[face] = details

    rebalance_color_counts(labels_by_face, details_by_face)
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

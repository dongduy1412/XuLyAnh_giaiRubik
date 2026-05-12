from pathlib import Path

import cv2
import numpy as np

from .rubik_state import FACE_ORDER


FACE_COLORS_BGR = {
    "U": (245, 245, 245),
    "R": (0, 0, 220),
    "F": (0, 170, 0),
    "D": (0, 230, 230),
    "L": (0, 120, 255),
    "B": (220, 70, 0),
}


def draw_face(labels: list[str], cell_size: int = 70, gap: int = 3) -> np.ndarray:
    if len(labels) != 9:
        raise ValueError("A Rubik face must contain 9 labels")

    side = cell_size * 3 + gap * 4
    image = np.full((side, side, 3), 35, dtype=np.uint8)
    for index, label in enumerate(labels):
        row, col = divmod(index, 3)
        x0 = gap + col * (cell_size + gap)
        y0 = gap + row * (cell_size + gap)
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        color = FACE_COLORS_BGR.get(label, (128, 128, 128))
        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 2)
        cv2.putText(image, label, (x0 + 22, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return image


def draw_unfolded_cube(labels_by_face: dict[str, list[str]], cell_size: int = 70) -> np.ndarray:
    face_images = {face: draw_face(labels_by_face[face], cell_size) for face in FACE_ORDER}
    face_side = next(iter(face_images.values())).shape[0]
    margin = 12
    canvas_height = face_side * 4 + margin * 5
    canvas_width = face_side * 3 + margin * 4
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

    positions = {
        "U": (1, 0),
        "L": (0, 1),
        "F": (1, 1),
        "R": (2, 1),
        "D": (1, 2),
        "B": (1, 3),
    }
    for face, (grid_x, grid_y) in positions.items():
        x0 = margin + grid_x * (face_side + margin)
        y0 = margin + grid_y * (face_side + margin)
        canvas[y0:y0 + face_side, x0:x0 + face_side] = face_images[face]
        cv2.putText(canvas, face, (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    return canvas


def save_unfolded_cube(path: str | Path, labels_by_face: dict[str, list[str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = draw_unfolded_cube(labels_by_face)
    cv2.imwrite(str(output_path), image)

from pathlib import Path

import cv2
import numpy as np

from .rubik_state import FACE_ORDER


# Fallback colors (standard Rubik) — only used when no real reference is available
DEFAULT_COLORS_BGR = {
    "U": (245, 245, 245),
    "R": (0, 0, 220),
    "F": (0, 170, 0),
    "D": (0, 230, 230),
    "L": (0, 120, 255),
    "B": (220, 70, 0),
}


def lab_to_bgr(lab_values: np.ndarray) -> tuple[int, int, int]:
    """Convert a single Lab color to BGR tuple for display."""
    lab_pixel = np.uint8([[lab_values.astype(np.uint8)]])
    bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
    b, g, r = bgr_pixel[0, 0]
    return (int(b), int(g), int(r))


def build_color_map_from_references(references: dict[str, np.ndarray]) -> dict[str, tuple[int, int, int]]:
    """Convert Lab reference colors to BGR tuples for visualization."""
    color_map = dict(DEFAULT_COLORS_BGR)  # start with defaults
    for face, lab in references.items():
        if isinstance(lab, np.ndarray):
            color_map[face] = lab_to_bgr(lab)
        elif isinstance(lab, (list, tuple)) and len(lab) == 3:
            color_map[face] = lab_to_bgr(np.array(lab, dtype=np.float32))
    return color_map


def _text_color_for_bg(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choose black or white text based on background brightness."""
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_face(
    labels: list[str],
    cell_size: int = 70,
    gap: int = 3,
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    if len(labels) != 9:
        raise ValueError("A Rubik face must contain 9 labels")

    colors = color_map if color_map is not None else DEFAULT_COLORS_BGR
    side = cell_size * 3 + gap * 4
    image = np.full((side, side, 3), 35, dtype=np.uint8)
    for index, label in enumerate(labels):
        row, col = divmod(index, 3)
        x0 = gap + col * (cell_size + gap)
        y0 = gap + row * (cell_size + gap)
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        bg_color = colors.get(label, (128, 128, 128))
        cv2.rectangle(image, (x0, y0), (x1, y1), bg_color, -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 2)
        text_color = _text_color_for_bg(bg_color)
        cv2.putText(image, label, (x0 + 22, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    return image


def draw_unfolded_cube(
    labels_by_face: dict[str, list[str]],
    cell_size: int = 70,
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    face_images = {face: draw_face(labels_by_face[face], cell_size, color_map=color_map) for face in FACE_ORDER}
    face_side = next(iter(face_images.values())).shape[0]
    margin = 12
    canvas_height = face_side * 3 + margin * 4
    canvas_width = face_side * 4 + margin * 5
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

    positions = {
        "U": (1, 0),
        "L": (0, 1),
        "F": (1, 1),
        "R": (2, 1),
        "B": (3, 1),
        "D": (1, 2),
    }
    for face, (grid_x, grid_y) in positions.items():
        x0 = margin + grid_x * (face_side + margin)
        y0 = margin + grid_y * (face_side + margin)
        canvas[y0:y0 + face_side, x0:x0 + face_side] = face_images[face]
        cv2.putText(canvas, face, (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    return canvas


def save_unfolded_cube(
    path: str | Path,
    labels_by_face: dict[str, list[str]],
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = draw_unfolded_cube(labels_by_face, color_map=color_map)
    cv2.imwrite(str(output_path), image)

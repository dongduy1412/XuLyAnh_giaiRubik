from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .color_recognizer import get_mean_lab
from .rubik_state import FACE_ORDER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "Color"
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "data" / "rubik_color_profile.json"
CALIBRATION_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CENTER_CELL_INDEX = 4
TARGET_HUES = {
    "R": 0.0,
    "F": 60.0,
    "D": 30.0,
    "L": 18.0,
    "B": 110.0,
}


def build_center_cell_profile(cells_by_face: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Build reference colors from center cells of available faces.

    Only processes faces present in cells_by_face, so it works incrementally
    as the user uploads faces one by one.
    """
    references = {}
    for face in FACE_ORDER:
        cells = cells_by_face.get(face)
        if cells is None:
            continue
        if len(cells) != 9:
            raise ValueError(f"Face {face} must contain exactly 9 cells, got {len(cells)}")
        references[face] = get_mean_lab(cells[CENTER_CELL_INDEX])
    return references


def profile_to_json(profile: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in value]
        for key, value in profile.items()
    }


def _load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    with Image.open(image_path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _crop_center(image: np.ndarray, margin_ratio: float = 0.12) -> np.ndarray:
    height, width = image.shape[:2]
    margin_x = max(1, int(width * margin_ratio))
    margin_y = max(1, int(height * margin_ratio))
    crop = image[margin_y:height - margin_y, margin_x:width - margin_x]
    if crop.size == 0:
        return image
    return crop


def _sample_hsv(path: str | Path) -> np.ndarray:
    image = _load_image(path)
    crop = _crop_center(image)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)


def _sample_lab(path: str | Path) -> np.ndarray:
    image = _load_image(path)
    crop = _crop_center(image)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0)


def _hue_distance(hue_a: float, hue_b: float) -> float:
    delta = abs(hue_a - hue_b)
    return min(delta, 180.0 - delta)


def _sample_cost(sample: np.ndarray, face: str) -> float:
    hue, saturation, value = sample.astype(float)
    if face == "U":
        return saturation * 6.0 + max(0.0, 245.0 - value) * 0.7 + abs(hue - 0.0) * 0.1
    target_hue = TARGET_HUES[face]
    return _hue_distance(hue, target_hue) + max(0.0, 85.0 - saturation) * 0.2 + max(0.0, 115.0 - value) * 0.05


def _assign_samples_to_faces(samples: list[np.ndarray]) -> dict[str, np.ndarray]:
    if len(samples) != len(FACE_ORDER):
        raise ValueError(f"Expected {len(FACE_ORDER)} calibration images, got {len(samples)}")

    best_cost = float("inf")
    best_assignment: dict[str, np.ndarray] | None = None
    for permuted_samples in permutations(samples):
        cost = sum(_sample_cost(sample, face) for sample, face in zip(permuted_samples, FACE_ORDER, strict=True))
        if cost < best_cost:
            best_cost = cost
            best_assignment = {face: sample for face, sample in zip(FACE_ORDER, permuted_samples, strict=True)}

    if best_assignment is None:
        raise ValueError("Unable to assign calibration samples to cube faces")
    return best_assignment


def _collect_calibration_images(calibration_dir: Path) -> list[Path]:
    if not calibration_dir.exists():
        raise FileNotFoundError(f"Calibration folder not found: {calibration_dir}")
    images = sorted(
        path for path in calibration_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CALIBRATION_EXTENSIONS
    )
    if len(images) != len(FACE_ORDER):
        raise ValueError(f"Expected {len(FACE_ORDER)} calibration images in {calibration_dir}, found {len(images)}")
    return images


def build_reference_profile(calibration_dir: str | Path = DEFAULT_CALIBRATION_DIR) -> dict[str, dict[str, object]]:
    calibration_dir = Path(calibration_dir)
    image_paths = _collect_calibration_images(calibration_dir)
    samples_lab = [_sample_lab(path) for path in image_paths]

    return {
        f"C{index + 1}": {
            "lab": [float(value) for value in sample],
            "source": str(image_paths[index]),
        }
        for index, sample in enumerate(samples_lab)
    }


def save_reference_profile(profile: dict[str, dict[str, object]], profile_path: str | Path = DEFAULT_PROFILE_PATH) -> Path:
    profile_path = Path(profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as file:
        json.dump({"version": 3, "space": "LAB", "type": "palette", "colors": profile}, file, ensure_ascii=False, indent=2)
    return profile_path


def _profile_is_stale(profile_path: Path, calibration_dir: Path) -> bool:
    if not profile_path.exists():
        return True
    if not calibration_dir.exists():
        return False
    try:
        image_paths = _collect_calibration_images(calibration_dir)
    except (FileNotFoundError, ValueError):
        return False
    profile_mtime = profile_path.stat().st_mtime
    for image_path in image_paths:
        if image_path.stat().st_mtime > profile_mtime:
            return True
    return False


def load_reference_profile(
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
    calibration_dir: str | Path = DEFAULT_CALIBRATION_DIR,
) -> dict[str, np.ndarray]:
    profile_path = Path(profile_path)
    calibration_dir = Path(calibration_dir)

    payload = None
    if profile_path.exists():
        with profile_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

    if (
        _profile_is_stale(profile_path, calibration_dir)
        or payload is None
        or payload.get("version") != 3
        or payload.get("space") != "LAB"
        or payload.get("type") != "palette"
    ):
        profile = build_reference_profile(calibration_dir)
        save_reference_profile(profile, profile_path)
    else:
        profile = payload["colors"]

    return {
        color_key: np.array(profile[color_key]["lab"], dtype=np.float32)
        for color_key in profile
    }

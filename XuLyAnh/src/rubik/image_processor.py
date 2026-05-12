from pathlib import Path

import cv2
import numpy as np


def load_face_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def center_square_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def _square_crop_from_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], padding_ratio: float = 0.04) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, bbox_width, bbox_height = bbox
    side = int(max(bbox_width, bbox_height) * (1 + padding_ratio * 2))
    center_x = x + bbox_width // 2
    center_y = y + bbox_height // 2
    x0 = max(0, center_x - side // 2)
    y0 = max(0, center_y - side // 2)
    x1 = min(width, x0 + side)
    y1 = min(height, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return center_square_crop(image)
    return crop


def detect_face_crop(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    color_mask = ((saturation > 45) & (value > 70)).astype(np.uint8) * 255
    white_mask = ((saturation < 55) & (value > 120)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(color_mask, white_mask)

    min_dim = min(image.shape[:2])
    kernel_size = max(7, (min_dim // 35) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return center_square_crop(image)

    min_area = image.shape[0] * image.shape[1] * 0.03
    candidates = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not candidates:
        return center_square_crop(image)

    contour = max(candidates, key=cv2.contourArea)
    return _square_crop_from_bbox(image, cv2.boundingRect(contour))


def resize_square(image: np.ndarray, size: int = 600) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_l = clahe.apply(l_channel)
    normalized_lab = cv2.merge((normalized_l, a_channel, b_channel))
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)


def preprocess_face(image: np.ndarray, size: int = 600, equalize: bool = False) -> np.ndarray:
    face = detect_face_crop(image)
    face = resize_square(face, size)
    if equalize:
        face = normalize_lighting(face)
    return cv2.GaussianBlur(face, (3, 3), 0)

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_face_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    with Image.open(image_path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def center_square_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def _square_crop_from_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], padding_ratio: float = 0.10) -> np.ndarray:
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


def _touches_border(bbox: tuple[int, int, int, int], width: int, height: int, margin: int) -> bool:
    x, y, bbox_width, bbox_height = bbox
    return x <= margin or y <= margin or x + bbox_width >= width - margin or y + bbox_height >= height - margin


def _score_rubik_region(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    x, y, bbox_width, bbox_height = bbox
    if bbox_width == 0 or bbox_height == 0:
        return 0.0

    area = bbox_width * bbox_height
    image_area = width * height
    area_ratio = area / image_area
    aspect = bbox_width / bbox_height
    if area_ratio < 0.015 or area_ratio > 0.60 or aspect < 0.55 or aspect > 1.80:
        return 0.0

    square_score = 1.0 - abs(bbox_width - bbox_height) / max(bbox_width, bbox_height)
    center_x = x + bbox_width / 2
    center_y = y + bbox_height / 2
    center_distance = abs(center_x - width / 2) / width + abs(center_y - height / 2) / height
    border_penalty = 0.35 if _touches_border(bbox, width, height, margin=max(4, min(width, height) // 80)) else 1.0
    return area_ratio * square_score * (1.0 - min(center_distance, 0.75)) * border_penalty


def _build_sticker_mask(image: np.ndarray, close: bool = False) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    chroma = np.sqrt((channel_a.astype(np.float32) - 128) ** 2 + (channel_b.astype(np.float32) - 128) ** 2)
    mask = ((chroma > 35) & (lightness > 80)).astype(np.uint8) * 255
    min_dim = min(image.shape[:2])
    open_size = max(3, (min_dim // 180) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    if close:
        close_size = max(9, (min_dim // 30) | 1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return mask


def _find_sticker_bboxes(mask: np.ndarray, image_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    height, width = image_shape
    image_area = height * width
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        area = bbox_width * bbox_height
        if area < image_area * 0.0004 or area > image_area * 0.05:
            continue
        aspect = bbox_width / bbox_height if bbox_height else 0
        if aspect < 0.45 or aspect > 2.2:
            continue
        if bbox_width < 10 or bbox_height < 10:
            continue
        if _touches_border((x, y, bbox_width, bbox_height), width, height, margin=max(3, min(width, height) // 120)):
            continue
        bboxes.append((x, y, bbox_width, bbox_height))
    return bboxes


def _union_bboxes(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(x for x, _, _, _ in bboxes)
    y0 = min(y for _, y, _, _ in bboxes)
    x1 = max(x + width for x, _, width, _ in bboxes)
    y1 = max(y + height for _, y, _, height in bboxes)
    return x0, y0, x1 - x0, y1 - y0


def _cluster_sticker_bboxes(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(bboxes) < 3:
        return []

    sizes = np.array([max(width, height) for _, _, width, height in bboxes], dtype=np.float32)
    median_size = float(np.median(sizes))
    max_distance = max(60.0, median_size * 4.4)
    centers = [np.array((x + width / 2, y + height / 2), dtype=np.float32) for x, y, width, height in bboxes]

    best_cluster: list[tuple[int, int, int, int]] = []
    best_score = -1.0
    for center in centers:
        cluster = [bbox for bbox, other_center in zip(bboxes, centers, strict=True) if np.linalg.norm(other_center - center) <= max_distance]
        if len(cluster) < 3:
            continue
        union = _union_bboxes(cluster)
        score = len(cluster) + _score_rubik_region(union, (10_000, 10_000))
        if score > best_score:
            best_cluster = cluster
            best_score = score
    return best_cluster


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _warp_square_from_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    ordered = _order_points(box)
    (top_left, top_right, bottom_right, bottom_left) = ordered

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    side = int(max(width_a, width_b, height_a, height_b))
    if side < 20:
        return center_square_crop(image)

    destination = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (side, side))


def _detect_by_connected_region(image: np.ndarray) -> np.ndarray:
    mask = _build_sticker_mask(image, close=True)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored_regions = []
    image_shape = image.shape[:2]
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        score = _score_rubik_region(bbox, image_shape)
        if score > 0:
            scored_regions.append((score, contour))
    if not scored_regions:
        return center_square_crop(image)
    _, best_contour = max(scored_regions, key=lambda item: item[0])
    x, y, width, height = cv2.boundingRect(best_contour)
    area_ratio = (width * height) / (image_shape[0] * image_shape[1])
    if area_ratio < 0.25:
        return center_square_crop(image)
    return _warp_square_from_contour(image, best_contour)


def _detect_by_dark_border(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_size = max(5, (min(image.shape[:2]) // 120) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    _, mask = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY_INV)
    kernel_size = max(7, (min(image.shape[:2]) // 45) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_shape = image.shape[:2]
    best_contour = None
    best_score = 0.0
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        score = _score_rubik_region(bbox, image_shape)
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return center_square_crop(image)

    x, y, width, height = cv2.boundingRect(best_contour)
    area_ratio = (width * height) / (image_shape[0] * image_shape[1])
    if area_ratio < 0.08:
        return center_square_crop(image)
    return _square_crop_from_bbox(image, (x, y, width, height), padding_ratio=0.03)


def detect_face_crop(image: np.ndarray) -> np.ndarray:
    border_crop = _detect_by_dark_border(image)
    if border_crop.shape[0] < image.shape[0] or border_crop.shape[1] < image.shape[1]:
        return border_crop

    mask = _build_sticker_mask(image, close=False)
    sticker_bboxes = _find_sticker_bboxes(mask, image.shape[:2])
    sticker_cluster = _cluster_sticker_bboxes(sticker_bboxes)
    if len(sticker_cluster) >= 6:
        return _square_crop_from_bbox(image, _union_bboxes(sticker_cluster), padding_ratio=0.06)
    return _detect_by_connected_region(image)


def _detect_rubik_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect the bounding box of the Rubik face using brightness thresholding.

    Works best when the Rubik cube is photographed against a dark background.
    Returns (x, y, w, h) or None if no suitable region is found.
    """
    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_size = max(5, min(h_img, w_img) // 80) | 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_size = max(15, min(h_img, w_img) // 20) | 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    open_size = max(3, min(h_img, w_img) // 150) | 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_score = 0.0
    best_bbox: tuple[int, int, int, int] | None = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / img_area
        if area_ratio < 0.05 or area_ratio > 0.95:
            continue

        aspect = w / h if h > 0 else 0
        if aspect < 0.4 or aspect > 2.5:
            continue

        squareness = 1.0 - abs(w - h) / max(w, h)
        cx = (x + w / 2) / w_img
        cy = (y + h / 2) / h_img
        center_score = 1.0 - (abs(cx - 0.5) + abs(cy - 0.5))

        score = area_ratio * squareness * max(center_score, 0.1)
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    return best_bbox


def smart_crop(image: np.ndarray, padding_ratio: float = 0.04) -> np.ndarray:
    """Detect the Rubik face in the image and crop tightly around it.

    Uses brightness-based thresholding to locate the Rubik cube region,
    then returns a square crop centered on that region.  Falls back to
    center_square_crop when detection fails.
    """
    bbox = _detect_rubik_bbox(image)
    if bbox is None:
        return center_square_crop(image)

    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]

    side = max(w, h)
    padding = int(side * padding_ratio)
    side += padding * 2

    cx = x + w // 2
    cy = y + h // 2

    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w_img, x0 + side)
    y1 = min(h_img, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return center_square_crop(image)
    return crop


def resize_square(image: np.ndarray, size: int = 600) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_l = clahe.apply(l_channel)
    normalized_lab = cv2.merge((normalized_l, a_channel, b_channel))
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)


def preprocess_face(image: np.ndarray, size: int = 600, equalize: bool = False, auto_crop: bool = False) -> np.ndarray:
    if auto_crop:
        face = detect_face_crop(image)
    else:
        face = smart_crop(image)
    face = resize_square(face, size)
    if equalize:
        face = normalize_lighting(face)
    return cv2.GaussianBlur(face, (3, 3), 0)

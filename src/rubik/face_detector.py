import cv2
import numpy as np


def _build_sticker_mask(face_image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    chroma = np.sqrt((channel_a.astype(np.float32) - 128) ** 2 + (channel_b.astype(np.float32) - 128) ** 2)
    mask = ((lightness > 75) & ((chroma > 14) | (lightness > 135))).astype(np.uint8) * 255
    min_dim = min(face_image.shape[:2])
    kernel_size = max(3, (min_dim // 120) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _find_sticker_bboxes(face_image: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = face_image.shape[:2]
    image_area = height * width
    min_side = min(height, width)
    mask = _build_sticker_mask(face_image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        area = bbox_width * bbox_height
        area_ratio = area / image_area
        if area_ratio < 0.008 or area_ratio > 0.20:
            continue
        if bbox_width < min_side * 0.10 or bbox_height < min_side * 0.10:
            continue
        aspect = bbox_width / bbox_height if bbox_height else 0
        if aspect < 0.45 or aspect > 2.4:
            continue
        bboxes.append((x, y, bbox_width, bbox_height))
    return bboxes


def _center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, width, height = bbox
    return x + width / 2, y + height / 2


def _select_grid_bboxes(bboxes: list[tuple[int, int, int, int]], image_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    if len(bboxes) < 9:
        return []

    height, width = image_shape
    center_x = width / 2
    center_y = height / 2

    def score(bbox: tuple[int, int, int, int]) -> float:
        x, y, bbox_width, bbox_height = bbox
        bbox_center_x, bbox_center_y = _center_of_bbox(bbox)
        area = bbox_width * bbox_height
        center_penalty = abs(bbox_center_x - center_x) / width + abs(bbox_center_y - center_y) / height
        return area * (1.0 - min(center_penalty, 0.8))

    selected = sorted(bboxes, key=score, reverse=True)[:9]
    selected = sorted(selected, key=lambda bbox: _center_of_bbox(bbox)[1])
    rows = [selected[index:index + 3] for index in range(0, 9, 3)]
    if any(len(row) != 3 for row in rows):
        return []
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda bbox: _center_of_bbox(bbox)[0]))
    return ordered


def _crop_bbox(face_image: np.ndarray, bbox: tuple[int, int, int, int], shrink_ratio: float = 0.10) -> np.ndarray:
    x, y, width, height = bbox
    shrink_x = int(width * shrink_ratio)
    shrink_y = int(height * shrink_ratio)
    x0 = x + shrink_x
    y0 = y + shrink_y
    x1 = x + width - shrink_x
    y1 = y + height - shrink_y
    crop = face_image[y0:y1, x0:x1]
    if crop.size == 0:
        return face_image[y:y + height, x:x + width]
    return crop


def split_face_into_cells(face_image: np.ndarray, grid_size: int = 3) -> list[np.ndarray]:
    if grid_size == 3:
        sticker_bboxes = _select_grid_bboxes(_find_sticker_bboxes(face_image), face_image.shape[:2])
        if len(sticker_bboxes) == 9:
            return [_crop_bbox(face_image, bbox) for bbox in sticker_bboxes]

    height, width = face_image.shape[:2]
    x_bounds = [round(width * index / grid_size) for index in range(grid_size + 1)]
    y_bounds = [round(height * index / grid_size) for index in range(grid_size + 1)]
    cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = y_bounds[row]
            y1 = y_bounds[row + 1]
            x0 = x_bounds[col]
            x1 = x_bounds[col + 1]
            cells.append(face_image[y0:y1, x0:x1])
    return cells


def inner_cell_crop(cell: np.ndarray, margin_ratio: float = 0.25) -> np.ndarray:
    height, width = cell.shape[:2]
    margin_y = int(height * margin_ratio)
    margin_x = int(width * margin_ratio)
    if margin_y * 2 >= height or margin_x * 2 >= width:
        return cell
    return cell[margin_y:height - margin_y, margin_x:width - margin_x]


def draw_grid(face_image: np.ndarray, grid_size: int = 3) -> np.ndarray:
    result = face_image.copy()
    if grid_size == 3:
        sticker_bboxes = _select_grid_bboxes(_find_sticker_bboxes(face_image), face_image.shape[:2])
        if len(sticker_bboxes) == 9:
            for index, (x, y, width, height) in enumerate(sticker_bboxes):
                cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
                center_x = x + width // 2
                center_y = y + height // 2
                cv2.circle(result, (center_x, center_y), 4, (0, 255, 0), -1)
                cv2.putText(result, str(index + 1), (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return result

    height, width = result.shape[:2]
    for index in range(1, grid_size):
        x = round(width * index / grid_size)
        y = round(height * index / grid_size)
        cv2.line(result, (x, 0), (x, height - 1), (0, 255, 0), 2)
        cv2.line(result, (0, y), (width - 1, y), (0, 255, 0), 2)
    return result

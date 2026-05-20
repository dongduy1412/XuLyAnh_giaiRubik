import cv2
import numpy as np


def split_face_into_cells(face_image: np.ndarray, grid_size: int = 3) -> list[np.ndarray]:
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
    height, width = result.shape[:2]
    for index in range(1, grid_size):
        x = round(width * index / grid_size)
        y = round(height * index / grid_size)
        cv2.line(result, (x, 0), (x, height - 1), (0, 255, 0), 2)
        cv2.line(result, (0, y), (width - 1, y), (0, 255, 0), 2)
    return result

from __future__ import annotations

import cv2
import numpy as np


def clahe_enhance(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray)


def denoise(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def threshold_defect(gray: np.ndarray) -> np.ndarray:
    enhanced = clahe_enhance(gray)
    blurred = denoise(enhanced)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def contour_boxes(gray: np.ndarray, min_area: int = 10):
    binary = threshold_defect(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h, float(area)))
    return boxes


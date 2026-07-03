from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog


def resize_gray(gray: np.ndarray, size: int = 128) -> np.ndarray:
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def hog_features(gray: np.ndarray) -> np.ndarray:
    image = resize_gray(gray, 128)
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def glcm_features(gray: np.ndarray) -> np.ndarray:
    image = resize_gray(gray, 128)
    quantized = (image / 32).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.concatenate([graycoprops(matrix, prop).ravel() for prop in props]).astype(np.float32)


def sift_bow_features(gray: np.ndarray, vocabulary: np.ndarray | None = None) -> np.ndarray:
    sift = cv2.SIFT_create()
    image = resize_gray(gray, 128)
    _, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = np.zeros((1, 128), dtype=np.float32)
    if vocabulary is None:
        return descriptors.mean(axis=0).astype(np.float32)
    distances = np.linalg.norm(descriptors[:, None, :] - vocabulary[None, :, :], axis=2)
    words = distances.argmin(axis=1)
    hist = np.bincount(words, minlength=len(vocabulary)).astype(np.float32)
    return hist / max(hist.sum(), 1.0)


def combined_features(gray: np.ndarray, use_sift: bool = True) -> np.ndarray:
    parts = [hog_features(gray), glcm_features(gray)]
    if use_sift:
        parts.append(sift_bow_features(gray))
    return np.concatenate(parts).astype(np.float32)


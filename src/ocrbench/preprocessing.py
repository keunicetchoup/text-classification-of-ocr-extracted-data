"""Image preprocessing variants, kept small and explicit so their effect can be measured.

Fix vs. thesis notebooks: one notebook used cv2.medianBlur(img, 1), which is a no-op.
All variants now live here and are selected by name in the OCR runner.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _row_score(binary: np.ndarray, angle: float) -> float:
    """Sharpness of the horizontal projection profile after rotating by `angle` degrees."""
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)
    prof = rot.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(prof) ** 2))


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Angle (degrees, cv2 convention) to rotate the page by so text lines become horizontal.

    Projection-profile method: text lines make the row histogram sharpest when horizontal.
    It replaces the earlier cv2.minAreaRect on all dark pixels, which measured the bounding
    box of the page content (and depends on the OpenCV angle convention), not the text skew.
    """
    scale = 1000.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    binary = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1].astype(np.uint8)
    if binary.sum() < 100:
        return 0.0
    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.5)
    best = max(coarse, key=lambda a: _row_score(binary, a))
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    return float(max(fine, key=lambda a: _row_score(binary, a)))


def deskew(gray: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Rotate the page so that text lines are horizontal (skew within +/- max_angle degrees)."""
    angle = estimate_skew(gray, max_angle)
    if abs(angle) < 0.2:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def otsu(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def denoise(img: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, k)


def clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


VARIANTS = {
    "raw":            lambda im: to_gray(im),
    "thesis":         lambda im: denoise(otsu(to_gray(im)), 3),             # grayscale + Otsu + median 3
    "deskew_otsu":    lambda im: denoise(otsu(deskew(to_gray(im))), 3),
    "clahe_adaptive": lambda im: denoise(adaptive(clahe(deskew(to_gray(im)))), 3),
}


def preprocess(img: np.ndarray, variant: str = "thesis") -> np.ndarray:
    return VARIANTS[variant](img)

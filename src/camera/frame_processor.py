from __future__ import annotations

import cv2 as cv
from typing import Tuple


def apply_clahe_lab(image: cv.typing.MatLike, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> cv.typing.MatLike:
    """Enhance contrast using CLAHE on the L channel in LAB color space.

    Returns a BGR image with the enhanced L channel.
    """
    if len(image.shape) != 3:
        return image

    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enh = clahe.apply(l)
    lab_enh = cv.merge((l_enh, a, b))
    return cv.cvtColor(lab_enh, cv.COLOR_LAB2BGR)


def to_grayscale(frame: cv.typing.MatLike, use_clahe: bool = False, clahe_clip: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> cv.typing.MatLike:
    """Convert an image to grayscale, optionally applying LAB+CLAHE first."""
    if len(frame.shape) == 3:
        img = frame
        if use_clahe:
            img = apply_clahe_lab(img, clip_limit=clahe_clip, tile_grid_size=tile_grid_size)
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return frame


def to_gray_blur(frame: cv.typing.MatLike, blur_kernel: int = 5, use_clahe: bool = False) -> cv.typing.MatLike:
    gray = to_grayscale(frame, use_clahe=use_clahe)
    kernel = max(3, blur_kernel)
    if kernel % 2 == 0:
        kernel += 1
    return cv.GaussianBlur(gray, (kernel, kernel), 0)


def center_crop(image: cv.typing.MatLike, margin_ratio: float = 0.15) -> cv.typing.MatLike:
    h, w = image.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x0, x1 = mx, max(mx + 1, w - mx)
    y0, y1 = my, max(my + 1, h - my)
    return image[y0:y1, x0:x1]

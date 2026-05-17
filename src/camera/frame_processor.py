from __future__ import annotations

import cv2 as cv


def to_gray_blur(frame: cv.typing.MatLike, blur_kernel: int = 5) -> cv.typing.MatLike:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
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

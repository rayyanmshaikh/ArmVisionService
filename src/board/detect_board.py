from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from src.camera.frame_processor import to_grayscale


@dataclass(slots=True)
class BoardCalibration:
    homography: np.ndarray
    inverse_homography: np.ndarray
    board_size: int
    outer_corners: np.ndarray
    internal_corners: np.ndarray


def _find_internal_corners(gray: np.ndarray, pattern_size: tuple[int, int] = (7, 7)) -> np.ndarray | None:
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
    found, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(pattern_size[1], pattern_size[0], 2)


def _estimate_outer_corners(internal: np.ndarray) -> np.ndarray:
    top_left = internal[0, 0]
    top_right = internal[0, -1]
    bottom_left = internal[-1, 0]
    bottom_right = internal[-1, -1]

    vx_top = (top_right - top_left) / 6.0
    vx_bottom = (bottom_right - bottom_left) / 6.0
    vy_left = (bottom_left - top_left) / 6.0
    vy_right = (bottom_right - top_right) / 6.0

    outer_tl = top_left - 1.0 * vx_top - 1.0 * vy_left
    outer_tr = top_right + 1.0 * vx_top - 1.0 * vy_right
    outer_br = bottom_right + 1.0 * vx_bottom + 1.0 * vy_right
    outer_bl = bottom_left - 1.0 * vx_bottom + 1.0 * vy_left

    return np.array([outer_tl, outer_tr, outer_br, outer_bl], dtype=np.float32)


def calibrate_from_frame(frame: np.ndarray, board_size: int = 800) -> BoardCalibration | None:
    gray = to_grayscale(frame, use_clahe=False) if len(frame.shape) == 3 else frame
    internal = _find_internal_corners(gray)
    if internal is None:
        return None

    outer = _estimate_outer_corners(internal)
    dst = np.array(
        [[0, 0], [board_size - 1, 0], [board_size - 1, board_size - 1], [0, board_size - 1]],
        dtype=np.float32,
    )

    homography = cv.getPerspectiveTransform(outer, dst)
    inverse_homography = cv.getPerspectiveTransform(dst, outer)
    return BoardCalibration(
        homography=homography,
        inverse_homography=inverse_homography,
        board_size=board_size,
        outer_corners=outer,
        internal_corners=internal,
    )


def warp_frame(frame: np.ndarray, calibration: BoardCalibration) -> np.ndarray:
    return cv.warpPerspective(frame, calibration.homography, (calibration.board_size, calibration.board_size))

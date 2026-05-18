from __future__ import annotations

import numpy as np

from src.board.detect_board import calibrate_from_frame, warp_frame
from src.board.grid_mapper import build_cell_regions


def test_detect_board_top_down(top_down_board_image: np.ndarray) -> None:
    calibration = calibrate_from_frame(top_down_board_image, board_size=800)
    assert calibration is not None
    warped = warp_frame(top_down_board_image, calibration)
    assert warped.shape[:2] == (800, 800)

    cells = build_cell_regions(800, orientation="white")
    assert len(cells) == 64
    assert cells[0].name == "a8"
    assert cells[-1].name == "h1"


def test_detect_board_angled(angled_board_image: np.ndarray) -> None:
    calibration = calibrate_from_frame(angled_board_image, board_size=800)
    assert calibration is not None


def test_no_board_returns_none(no_board_image: np.ndarray) -> None:
    calibration = calibrate_from_frame(no_board_image, board_size=800)
    assert calibration is None

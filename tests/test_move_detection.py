from __future__ import annotations

import numpy as np

from src.board.detect_board import calibrate_from_frame, warp_frame
from src.board.grid_mapper import build_cell_regions
from src.board.move_detector import MoveDetector


def test_detect_move_e2e4(e2e4_pair: tuple[np.ndarray, np.ndarray]) -> None:
    before, after = e2e4_pair
    detector = MoveDetector(change_threshold=5.0)
    cells = build_cell_regions(board_size=800, orientation="white")

    move = detector.detect_move(before, after, cells)
    assert move == "e2e4"


def test_detect_no_move(top_down_board_image: np.ndarray) -> None:
    detector = MoveDetector(change_threshold=5.0)
    cells = build_cell_regions(board_size=800, orientation="white")
    move = detector.detect_move(top_down_board_image, top_down_board_image.copy(), cells)
    assert move is None


def test_detect_move_full_board_regression(
    top_down_board_image: np.ndarray,
    full_board_e2e4_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    calibration = calibrate_from_frame(top_down_board_image, board_size=800)
    assert calibration is not None

    before, after = full_board_e2e4_pair
    warped_before = warp_frame(before, calibration)
    warped_after = warp_frame(after, calibration)

    detector = MoveDetector(change_threshold=5.0)
    cells = build_cell_regions(board_size=800, orientation="white")

    move = detector.detect_move(warped_before, warped_after, cells)
    assert move == "e2e4"

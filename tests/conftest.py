from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from src.board.grid_mapper import build_cell_regions


def generate_chessboard(board_size: int = 800) -> np.ndarray:
    cell = board_size // 8
    board = np.zeros((board_size, board_size, 3), dtype=np.uint8)
    light = (235, 235, 235)
    dark = (40, 95, 40)

    for row in range(8):
        for col in range(8):
            color = light if (row + col) % 2 == 0 else dark
            x0, y0 = col * cell, row * cell
            x1, y1 = (col + 1) * cell, (row + 1) * cell
            cv.rectangle(board, (x0, y0), (x1, y1), color, thickness=-1)

    return board


def perspective_tilt(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[70, 40], [w - 90, 20], [w - 40, h - 60], [30, h - 30]])
    matrix = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(image, matrix, (w, h))


def square_center(square: str, board_size: int = 800) -> tuple[int, int, int]:
    cells = build_cell_regions(board_size=board_size, orientation="white")
    cell_by_name = {cell.name: cell for cell in cells}
    target = cell_by_name[square]
    cx = (target.x0 + target.x1) // 2
    cy = (target.y0 + target.y1) // 2
    radius = max(8, (target.x1 - target.x0) // 4)
    return cx, cy, radius


def draw_piece(board: np.ndarray, square: str, color: tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    image = board.copy()
    cx, cy, radius = square_center(square)
    cv.circle(image, (cx, cy), radius, color, thickness=-1)
    return image


def draw_pieces(board: np.ndarray, squares: list[str], color: tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    image = board.copy()
    for square in squares:
        cx, cy, radius = square_center(square)
        cv.circle(image, (cx, cy), radius, color, thickness=-1)
    return image


def starting_position_squares() -> list[str]:
    return [
        "a1",
        "b1",
        "c1",
        "d1",
        "e1",
        "f1",
        "g1",
        "h1",
        "a2",
        "b2",
        "c2",
        "d2",
        "e2",
        "f2",
        "g2",
        "h2",
        "a7",
        "b7",
        "c7",
        "d7",
        "e7",
        "f7",
        "g7",
        "h7",
        "a8",
        "b8",
        "c8",
        "d8",
        "e8",
        "f8",
        "g8",
        "h8",
    ]


@pytest.fixture(scope="session")
def top_down_board_image() -> np.ndarray:
    return generate_chessboard()


@pytest.fixture(scope="session")
def angled_board_image(top_down_board_image: np.ndarray) -> np.ndarray:
    return perspective_tilt(top_down_board_image)


@pytest.fixture(scope="session")
def no_board_image() -> np.ndarray:
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    cv.circle(image, (400, 400), 220, (180, 30, 120), thickness=-1)
    return image


@pytest.fixture(scope="session")
def e2e4_pair(top_down_board_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    before = draw_piece(top_down_board_image, "e2")
    after = draw_piece(top_down_board_image, "e4")
    return before, after


@pytest.fixture(scope="session")
def full_board_image(top_down_board_image: np.ndarray) -> np.ndarray:
    return draw_pieces(top_down_board_image, starting_position_squares())


@pytest.fixture(scope="session")
def full_board_e2e4_pair(full_board_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    before = full_board_image
    after = draw_pieces(
        generate_chessboard(),
        [square for square in starting_position_squares() if square != "e2"],
    )
    after = draw_piece(after, "e4")
    return before, after


@pytest.fixture(scope="session")
def fixtures_manifest() -> dict[str, object]:
    manifest_path = Path(__file__).parent / "fixtures" / "fixtures_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

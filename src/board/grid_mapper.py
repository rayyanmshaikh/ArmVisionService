from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CellRegion:
    name: str
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int


def _cell_name(row: int, col: int, orientation: str = "white") -> str:
    orientation = orientation.lower()
    files = "abcdefgh"
    if orientation == "black":
        file_char = files[::-1][col]
        rank = row + 1
    else:
        file_char = files[col]
        rank = 8 - row
    return f"{file_char}{rank}"


def build_cell_regions(board_size: int = 800, orientation: str = "white") -> list[CellRegion]:
    cell_size = board_size / 8.0
    cells: list[CellRegion] = []
    for row in range(8):
        for col in range(8):
            x0 = int(round(col * cell_size))
            y0 = int(round(row * cell_size))
            x1 = int(round((col + 1) * cell_size))
            y1 = int(round((row + 1) * cell_size))
            cells.append(CellRegion(_cell_name(row, col, orientation), row, col, x0, y0, x1, y1))
    return cells

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.board.grid_mapper import CellRegion
from src.camera.frame_processor import center_crop


@dataclass(frozen=True, slots=True)
class CellMetrics:
    square: str
    diff_score: float
    occupancy_before: float
    occupancy_after: float


class MoveDetector:
    def __init__(
        self,
        change_threshold: float = 10.0,
        max_candidates: int = 2,
        secondary_factor: float = 0.6,
    ) -> None:

        self.change_threshold = change_threshold
        self.max_candidates = max_candidates
        self.secondary_factor = secondary_factor

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    @staticmethod
    def _occupancy_score(cell: np.ndarray) -> float:
        cropped = center_crop(cell, margin_ratio=0.2)
        lap = cv.Laplacian(cropped, cv.CV_32F)
        return float(np.var(lap))

    def _metric_for_cell(self, before: np.ndarray, after: np.ndarray, cell: CellRegion) -> CellMetrics:
        before_cell = before[cell.y0 : cell.y1, cell.x0 : cell.x1]
        after_cell = after[cell.y0 : cell.y1, cell.x0 : cell.x1]

        diff = cv.absdiff(before_cell, after_cell)
        diff_score = float(np.mean(diff))

        return CellMetrics(
            square=cell.name,
            diff_score=diff_score,
            occupancy_before=self._occupancy_score(before_cell),
            occupancy_after=self._occupancy_score(after_cell),
        )

    def detect_move(self, before: np.ndarray, after: np.ndarray, cells: list[CellRegion]) -> str | None:
        before_gray = self._to_gray(before)
        after_gray = self._to_gray(after)

        metrics = [self._metric_for_cell(before_gray, after_gray, cell) for cell in cells]

        primary_changed = [m for m in metrics if m.diff_score >= self.change_threshold]

        if len(primary_changed) >= self.max_candidates:
            top = sorted(primary_changed, key=lambda metric: metric.diff_score, reverse=True)[: self.max_candidates]
        else:
            sorted_all = sorted(metrics, key=lambda metric: metric.diff_score, reverse=True)
            top = sorted_all[: self.max_candidates]

            if len(top) < 2 or top[1].diff_score < (self.change_threshold * self.secondary_factor):
                return None

        deltas = [(metric.square, metric.occupancy_after - metric.occupancy_before) for metric in top]
        deltas_sorted = sorted(deltas, key=lambda item: item[1])

        from_square = deltas_sorted[0][0]
        to_square = deltas_sorted[-1][0]

        if from_square == to_square:
            return None

        return f"{from_square}{to_square}"

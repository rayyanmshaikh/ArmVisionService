from __future__ import annotations

import numpy as np
import cv2 as cv
from pathlib import Path


class CameraStream:
    def __init__(self, camera_index: int | None = None, auto: bool = True, probe_max: int = 6, calib_path: str | None = None, undistort: bool = False) -> None:
        self.camera_index = camera_index
        self.auto = auto
        self.probe_max = probe_max
        self.capture: cv.VideoCapture | None = None
        self.selected_index: int | None = None

        # calibration / undistort
        self.calib_path = calib_path
        self.undistort = undistort
        self._do_undistort = False
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None

    def _load_calibration_for_size(self, frame_size: tuple[int, int]) -> None:
        # try provided path, otherwise default to repo config
        calib_file = None
        if self.calib_path:
            p = Path(self.calib_path)
            if p.exists():
                calib_file = p

        if calib_file is None:
            repo_root = Path(__file__).resolve().parents[2]
            default = repo_root / "config" / "camera_calib.npz"
            if default.exists():
                calib_file = default

        if calib_file is None or not self.undistort:
            self._do_undistort = False
            return

        data = np.load(str(calib_file))
        if "mtx" not in data or "dist" not in data:
            self._do_undistort = False
            return

        mtx = data["mtx"]
        dist = data["dist"]

        w, h = frame_size
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        map1, map2 = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv.CV_16SC2)
        self._map1 = map1
        self._map2 = map2
        self._do_undistort = True

    def open(self) -> bool:
        if self.camera_index is not None:
            self.capture = cv.VideoCapture(self.camera_index)
            if self.capture.isOpened():
                # determine frame size and load calibration maps if available
                w = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
                if w > 0 and h > 0:
                    self._load_calibration_for_size((w, h))
                self.selected_index = self.camera_index
                return True
            self.capture.release()
            self.capture = None
            return False

        if not self.auto:
            return False

        for index in range(self.probe_max):
            cap = cv.VideoCapture(index)
            if cap.isOpened():
                w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
                if w > 0 and h > 0:
                    self._load_calibration_for_size((w, h))
                self.capture = cap
                self.selected_index = index
                return True
            cap.release()
        return False

    def read(self) -> tuple[bool, cv.typing.MatLike | None]:
        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None

        if self._do_undistort and self._map1 is not None and self._map2 is not None:
            try:
                frame = cv.remap(frame, self._map1, self._map2, interpolation=cv.INTER_LINEAR)
            except Exception:
                # fallback to original frame if remap fails
                pass

        return ok, frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

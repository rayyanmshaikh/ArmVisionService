from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from enum import Enum
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.board.detect_board import BoardCalibration, calibrate_from_frame, warp_frame
from src.board.grid_mapper import build_cell_regions
from src.board.move_detector import MoveDetector
from src.camera.camera_stream import CameraStream
from src.camera.frame_processor import to_gray_blur
from src.utils.config_loader import ServiceConfig


class RunState(Enum):
    WAITING_TO_CAPTURE_BASELINE = "waiting_to_capture_baseline"
    WAITING_FOR_ROBOT = "waiting_for_robot"
    HUMAN_TURN = "human_turn"
    CAPTURE_MOVE = "capture_move"


class VisionRuntime:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("vision-runtime")
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._last_move: str | None = None
        self._calibrated = False
        self._camera_index: int | None = None
        self._state: RunState = RunState.WAITING_TO_CAPTURE_BASELINE
        self._baseline_warp: object | None = None
        self._current_warp: object | None = None

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return True

    def stop(self) -> bool:
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                return False
            self._stop_event.set()
            self._thread.join(timeout=3.0)
            return True

    def state(self) -> dict[str, object]:
        running = bool(self._thread and self._thread.is_alive())
        return {
            "running": running,
            "calibrated": self._calibrated,
            "camera_index": self._camera_index,
            "last_move": self._last_move,
            "vision_state": self._state.value,
            "config": asdict(self.config),
        }

    @property
    def last_move(self) -> str | None:
        return self._last_move

    def _run_loop(self) -> None:
        stream = CameraStream(
            camera_index=self.config.camera_index,
            auto=self.config.camera_auto,
            probe_max=self.config.camera_probe_max,
        )
        if not stream.open():
            self.logger.error("Unable to open camera. Check CAMERA_INDEX / camera permissions.")
            return

        self._camera_index = stream.selected_index
        self.logger.info("Camera opened: index=%s", self._camera_index)

        detector = MoveDetector(change_threshold=self.config.change_threshold)
        cells = build_cell_regions(self.config.warp_size, orientation=self.config.board_orientation)

        calibration: BoardCalibration | None = None

        try:
            while not self._stop_event.is_set():
                ok, frame = stream.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                if calibration is None:
                    calibration = calibrate_from_frame(frame, board_size=self.config.warp_size)
                    if calibration is None:
                        time.sleep(0.01)
                        continue
                    self._calibrated = True
                    self.logger.info("Board calibration succeeded.")
                    self.logger.info("Waiting for baseline capture. Call POST /capture-baseline after placing pieces.")
                    continue

                warped = warp_frame(frame, calibration)
                processed = to_gray_blur(warped)
                
                with self._lock:
                    self._current_warp = processed
                    state = self._state

                if state == RunState.WAITING_TO_CAPTURE_BASELINE:
                    time.sleep(0.05)
                elif state == RunState.HUMAN_TURN:
                    if self._baseline_warp is None:
                        self._baseline_warp = processed
                        print("Human turn: press SPACE when the move is complete.")
                elif state == RunState.CAPTURE_MOVE:
                    before = self._baseline_warp if self._baseline_warp is not None else processed
                    after = processed

                    for _ in range(3):
                        ok2, frame2 = stream.read()
                        if not ok2 or frame2 is None:
                            time.sleep(0.02)
                            continue
                        warped2 = warp_frame(frame2, calibration)
                        after = to_gray_blur(warped2)
                        time.sleep(0.02)

                    move = detector.detect_move(before, after, cells)
                    if move:
                        self._last_move = move
                        print(move, flush=True)
                        self._post_move(move)
                        with self._lock:
                            self._state = RunState.WAITING_FOR_ROBOT
                        self._baseline_warp = None
                        self.logger.info("Detected move: %s", move)
                    else:
                        self.logger.info("No valid move detected; staying on human turn")
                        with self._lock:
                            self._state = RunState.HUMAN_TURN
                        self._baseline_warp = before
                        print("No valid move detected. Please make a valid move and press SPACE again.")
                elif state == RunState.WAITING_FOR_ROBOT:
                    time.sleep(0.05)

                time.sleep(self.config.loop_sleep_ms / 1000.0)
        finally:
            stream.release()

    def on_human_move_complete(self) -> bool:
        with self._lock:
            if self._state != RunState.HUMAN_TURN:
                self.logger.info("on_human_move_complete ignored; state=%s", self._state)
                return False
            if self._baseline_warp is None:
                self.logger.info("on_human_move_complete ignored; baseline not ready yet")
                return False
            self._state = RunState.CAPTURE_MOVE
        self.logger.info("Human move complete: capturing before/after frames")
        print("Received human move signal — capturing move...")
        return True

    def on_capture_baseline(self) -> bool:
        with self._lock:
            if self._state != RunState.WAITING_TO_CAPTURE_BASELINE:
                self.logger.info("on_capture_baseline ignored; state=%s", self._state)
                return False
            if self._current_warp is None:
                self.logger.info("on_capture_baseline ignored; current frame not ready yet")
                return False
            self._baseline_warp = self._current_warp
            self._state = RunState.HUMAN_TURN
        self.logger.info("Baseline captured. Ready for human moves.")
        print("Baseline captured with pieces. Make your first move and call POST /human-done when complete.")
        return True

    def on_robot_done(self) -> bool:
        with self._lock:
            if self._state not in {RunState.WAITING_FOR_ROBOT, RunState.HUMAN_TURN}:
                self.logger.info("on_robot_done ignored; state=%s", self._state)
                return False
            self._state = RunState.HUMAN_TURN
            self._baseline_warp = self._current_warp
        self.logger.info("Robot finished; returning to HUMAN_TURN with updated baseline")
        print("Robot finished — human turn started. Make your next move and call POST /human-done when complete.")
        return True

    def _post_move(self, move: str) -> None:
        if not self.config.move_post_url:
            return

        payload = json.dumps({"move": move}).encode("utf-8")
        request = Request(
            self.config.move_post_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=5) as response:
                response.read()
        except URLError as exc:
            self.logger.warning("Failed to POST move to %s: %s", self.config.move_post_url, exc)
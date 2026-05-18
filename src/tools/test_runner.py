from __future__ import annotations

import time

import cv2 as cv

from src.board.detect_board import calibrate_from_frame, warp_frame
from src.board.grid_mapper import build_cell_regions
from src.board.move_detector import MoveDetector
from src.camera.camera_stream import CameraStream
from src.camera.frame_processor import to_gray_blur
from src.tools.visualizer import draw_calibration_overlay, draw_cell_notations
from src.utils.config_loader import ServiceConfig


def run_test_mode(config: ServiceConfig) -> None:
    stream = CameraStream(
        camera_index=config.camera_index,
        auto=config.camera_auto,
        probe_max=config.camera_probe_max,
    )

    if not stream.open():
        print("ERROR: Unable to open camera. Check CAMERA_INDEX / camera permissions.")
        return

    print(f"Camera opened: index={stream.selected_index}")
    print("Press SPACE to confirm the human move, 'q' to quit, 'r' to reset calibration")

    calibration = None
    baseline_gray = None
    last_move = None
    current_warp_gray = None

    cells = build_cell_regions(board_size=config.warp_size, orientation=config.board_orientation)
    detector = MoveDetector(change_threshold=config.change_threshold)

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if calibration is None:
                calibration = calibrate_from_frame(frame, board_size=config.warp_size)
                if calibration is not None:
                    print("Board calibration succeeded")
                    print("Place pieces for the first human turn, then press SPACE to capture a baseline image.")

            frame_with_calib = draw_calibration_overlay(frame, calibration)

            if calibration is not None:
                warped = warp_frame(frame, calibration)
                current_warp_gray = to_gray_blur(warped)
                warped = draw_cell_notations(warped, cells)

                status_text = "CALIBRATED | HUMAN TURN" if baseline_gray is not None else "CALIBRATED | PLACE PIECES (press SPACE to capture baseline)"
                cv.putText(
                    warped,
                    status_text,
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                if last_move:
                    cv.putText(warped, f"Last Move: {last_move}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv.imshow("Warped Board", warped)
            else:
                cv.putText(
                    frame_with_calib,
                    "Waiting for calibration... show empty board",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            if last_move:
                cv.putText(frame_with_calib, f"Last Move: {last_move}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status = "CALIBRATED" if calibration else "WAITING"
            bottom_msg = (
                "Press SPACE to capture baseline after placing pieces" if calibration and baseline_gray is None
                else "Press SPACE when the move is complete"
            )
            cv.putText(
                frame_with_calib,
                f"Status: {status} | {bottom_msg}",
                (10, frame_with_calib.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv.imshow("Raw Frame (Calibration)", frame_with_calib)

            key = cv.waitKey(30) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break
            if key == ord("r"):
                calibration = None
                baseline_gray = None
                current_warp_gray = None
                last_move = None
                print("Calibration reset. Show the board to recalibrate.")
            if key == ord(" "):
                if calibration is None or current_warp_gray is None:
                    print("Baseline not ready yet. Wait for calibration.")
                    continue

                # If we don't have a baseline yet, use SPACE to capture it (after the user placed pieces)
                if baseline_gray is None:
                    captured = None
                    for _ in range(5):
                        ok2, frame2 = stream.read()
                        if not ok2 or frame2 is None:
                            time.sleep(0.02)
                            continue
                        warped2 = warp_frame(frame2, calibration)
                        captured = to_gray_blur(warped2)
                        time.sleep(0.02)

                    if captured is not None:
                        baseline_gray = captured
                        print("Baseline captured. Now make your first human move and press SPACE to detect it.")
                    else:
                        print("Failed to capture baseline. Try again.")
                    continue

                # Baseline exists -> treat SPACE as 'human move complete' and detect change
                after_gray = current_warp_gray
                for _ in range(3):
                    ok2, frame2 = stream.read()
                    if not ok2 or frame2 is None:
                        time.sleep(0.02)
                        continue
                    warped2 = warp_frame(frame2, calibration)
                    after_gray = to_gray_blur(warped2)
                    time.sleep(0.02)

                move = detector.detect_move(baseline_gray, after_gray, cells)
                if move:
                    last_move = move
                    baseline_gray = after_gray
                    print(move, flush=True)
                    print("Test mode: re-armed for the next human move.")
                else:
                    print("No valid move detected. Please make a valid move and press SPACE again.")

    finally:
        stream.release()
        cv.destroyAllWindows()
        print("Test mode closed.")
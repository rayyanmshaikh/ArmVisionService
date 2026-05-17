from __future__ import annotations

import argparse
import time

import cv2 as cv
import numpy as np

from src.board.detect_board import calibrate_from_frame, warp_frame
from src.board.grid_mapper import build_cell_regions
from src.board.move_detector import MoveDetector
from src.camera.camera_stream import CameraStream
from src.camera.frame_processor import to_gray_blur


def draw_calibration_overlay(frame: np.ndarray, calibration) -> np.ndarray:
    """Draw the detected board corners and outer boundary on the raw frame."""
    output = frame.copy()
    if calibration is None:
        return output

    outer = calibration.outer_corners
    # draw outer boundary
    pts = np.int32(outer)
    cv.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # draw corner circles
    for pt in pts:
        cv.circle(output, tuple(pt), 5, (0, 255, 0), -1)

    # draw internal corners
    internal = calibration.internal_corners
    for row in internal:
        for corner in row:
            pt = tuple(np.int32(corner))
            cv.circle(output, pt, 2, (255, 0, 0), -1)

    return output


def draw_changed_squares(warped: np.ndarray, before_gray: np.ndarray, after_gray: np.ndarray, cells: list, threshold: float = 10.0) -> np.ndarray:
    """Highlight squares that changed significantly."""
    output = warped.copy()
    changed_list = []

    for cell in cells:
        b = before_gray[cell.y0:cell.y1, cell.x0:cell.x1]
        a = after_gray[cell.y0:cell.y1, cell.x0:cell.x1]
        diff = cv.absdiff(b, a)
        score = float(np.mean(diff))

        if score >= threshold:
            changed_list.append((cell, score))
            # red rectangle for changed squares
            cv.rectangle(output, (cell.x0, cell.y0), (cell.x1, cell.y1), (0, 0, 255), thickness=2)
            label = f"{cell.name}:{int(score)}"
            cv.putText(output, label, (cell.x0 + 4, cell.y0 + 16), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return output, changed_list


def run_visualizer(
    camera_index: int | None = None,
    camera_auto: bool = True,
    board_orientation: str = "white",
    threshold: float = 20.0,
    board_size: int = 800,
    probe_max: int = 6,
) -> None:
    """Run live visualization of board detection and move detection."""
    stream = CameraStream(camera_index=camera_index, auto=camera_auto, probe_max=probe_max)

    if not stream.open():
        print("ERROR: Unable to open camera. Check CAMERA_INDEX / camera permissions.")
        return

    print(f"Camera opened: index={stream.selected_index}")
    print("Press 'q' to quit, 'r' to reset calibration, 'c' to recalibrate on next stable board")

    calibration = None
    previous_warp_gray = None
    last_move = None
    last_move_time = 0.0
    move_cooldown_ms = 500.0
    consecutive_stable = 0
    last_stable_move = None

    cells = build_cell_regions(board_size=board_size, orientation=board_orientation)
    detector = MoveDetector(change_threshold=threshold)

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # try to calibrate if not yet done
            if calibration is None:
                calibration = calibrate_from_frame(frame, board_size=board_size)
                if calibration is not None:
                    print("✓ Board calibration succeeded")
                    warped = warp_frame(frame, calibration)
                    previous_warp_gray = to_gray_blur(warped)
                    consecutive_stable = 0
                    last_stable_move = None

            # draw calibration overlay on raw frame
            frame_with_calib = draw_calibration_overlay(frame, calibration)

            # if calibrated, warp and detect
            if calibration is not None:
                warped = warp_frame(frame, calibration)
                warped_gray = to_gray_blur(warped)

                # detect changes
                if previous_warp_gray is not None:
                    move = detector.detect_move(previous_warp_gray, warped_gray, cells)
                    now = time.time() * 1000.0
                    time_since = now - last_move_time

                    # debounce and stability check
                    if move and time_since >= move_cooldown_ms:
                        if move == last_stable_move:
                            consecutive_stable += 1
                        else:
                            consecutive_stable = 1
                            last_stable_move = move

                        if consecutive_stable >= 2 and move != last_move:
                            last_move = move
                            last_move_time = now
                            consecutive_stable = 0
                            print(f"MOVE: {move}")
                    else:
                        consecutive_stable = 0
                        last_stable_move = None

                    # highlight changed squares
                    warped_with_overlay, _ = draw_changed_squares(warped, previous_warp_gray, warped_gray, cells, threshold=threshold)
                else:
                    warped_with_overlay = warped.copy()

                previous_warp_gray = warped_gray

                # display the warped board with changed squares
                cv.imshow("Warped Board (Overlay)", warped_with_overlay)
            else:
                msg = "Waiting for calibration... show empty board"
                cv.putText(frame_with_calib, msg, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # display raw frame with calibration overlay
            if last_move:
                cv.putText(frame_with_calib, f"Last Move: {last_move}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status = "CALIBRATED" if calibration else "WAITING"
            cv.putText(frame_with_calib, f"Status: {status} | Threshold: {threshold}", (10, frame_with_calib.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv.imshow("Raw Frame (Calibration)", frame_with_calib)

            # keyboard control
            key = cv.waitKey(30) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break
            elif key == ord("r"):
                calibration = None
                previous_warp_gray = None
                consecutive_stable = 0
                last_stable_move = None
                print("Calibration reset. Show the board to recalibrate.")
            elif key == ord("c"):
                calibration = None
                print("Will recalibrate on next stable board view.")

    finally:
        stream.release()
        cv.destroyAllWindows()
        print("Visualizer closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live board detection and move visualization")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default: auto-detect)")
    parser.add_argument("--orientation", "-o", choices=["white", "black"], default="white", help="Board orientation")
    parser.add_argument("--threshold", "-t", type=float, default=20.0, help="Diff threshold for changed squares")
    parser.add_argument("--board-size", type=int, default=800, help="Warped board size in pixels")
    parser.add_argument("--probe-max", type=int, default=6, help="Maximum number of camera indices to probe")

    args = parser.parse_args()
    run_visualizer(
        camera_index=args.camera,
        board_orientation=args.orientation,
        threshold=args.threshold,
        board_size=args.board_size,
        probe_max=args.probe_max,
    )


if __name__ == "__main__":
    main()

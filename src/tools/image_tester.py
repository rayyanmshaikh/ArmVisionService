from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np

from src.board.detect_board import calibrate_from_frame, warp_frame
from src.board.grid_mapper import build_cell_regions
from src.board.move_detector import MoveDetector
from src.camera.frame_processor import to_grayscale


def load_images(dirpath: Path) -> List[Tuple[Path, np.ndarray]]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images: List[Tuple[Path, np.ndarray]] = []
    for p in sorted(dirpath.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            img = cv.imread(str(p))
            if img is None:
                continue
            images.append((p, img))
    return images


def find_calibration(images: List[Tuple[Path, np.ndarray]], calib_name: str | None = None) -> Tuple[Path, np.ndarray] | None:
    if calib_name:
        for p, img in images:
            if p.name == calib_name:
                return p, img

    # prefer files named like empty* or calib*
    for p, img in images:
        if p.stem.lower().startswith("empty") or p.stem.lower().startswith("calib"):
            return p, img

    # fallback to first image
    return images[0] if images else None


def grayscale_if_needed(img: np.ndarray) -> np.ndarray:
    return to_grayscale(img, use_clahe=True) if len(img.shape) == 3 else img


def analyze(dirpath: Path, calib_name: str | None, outdir: Path, diff_threshold: float = 10.0) -> None:
    images = load_images(dirpath)
    if not images:
        print(f"No images found in {dirpath}")
        return

    calib_item = find_calibration(images, calib_name)
    if calib_item is None:
        print("No calibration image available.")
        return

    calib_path, calib_img = calib_item
    print(f"Using calibration image: {calib_path.name}")

    calib = calibrate_from_frame(calib_img)
    if calib is None:
        print("Failed to detect chessboard calibration from the chosen image.")
        print("Try supplying a clearer calibration image (filename starting with 'empty' or 'calib').")
        return

    board_size = calib.board_size
    cells = build_cell_regions(board_size=board_size)

    warped_base = warp_frame(calib_img, calib)
    base_gray = grayscale_if_needed(warped_base)

    md = MoveDetector(change_threshold=diff_threshold)

    outdir.mkdir(parents=True, exist_ok=True)

    for p, img in images:
        if p == calib_path:
            continue
        warped = warp_frame(img, calib)
        warped_gray = grayscale_if_needed(warped)

        rows = []
        changed = []
        for cell in cells:
            before_cell = base_gray[cell.y0:cell.y1, cell.x0:cell.x1]
            after_cell = warped_gray[cell.y0:cell.y1, cell.x0:cell.x1]
            diff = cv.absdiff(before_cell, after_cell)
            diff_score = float(np.mean(diff))

            # occupancy measured via Laplacian variance
            cropped = after_cell
            lap = cv.Laplacian(cropped, cv.CV_32F)
            occupancy = float(np.var(lap))

            rows.append((cell.name, diff_score, occupancy))
            if diff_score >= diff_threshold:
                changed.append((cell.name, diff_score))

        # write CSV
        out_csv = outdir / f"{p.stem}_cells.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["square", "diff_score", "occupancy_after"]) 
            for r in rows:
                writer.writerow(r)

        print(f"Analyzed {p.name}: {len(changed)} changed squares (threshold={diff_threshold}). CSV -> {out_csv}")
        if changed:
            top = sorted(changed, key=lambda t: t[1], reverse=True)[:10]
            print("  Top changes:")
            for name, score in top:
                print(f"   - {name}: {score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test board images against detector calibration")
    parser.add_argument("--dir", "-d", required=True, help="Directory with images to analyze")
    parser.add_argument("--calib", "-c", required=False, help="Filename of calibration (empty) image")
    parser.add_argument("--out", "-o", required=False, default="image_test_results", help="Output directory for CSVs")
    parser.add_argument("--threshold", "-t", type=float, default=10.0, help="Diff threshold for changed squares")

    args = parser.parse_args()
    analyze(Path(args.dir), args.calib, Path(args.out), diff_threshold=args.threshold)


if __name__ == "__main__":
    main()

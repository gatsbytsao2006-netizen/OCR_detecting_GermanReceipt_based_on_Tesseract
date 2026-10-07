"""Preprocess a receipt photo and run Tesseract OCR on the cleaned image.

Pipeline: load -> perspective correction -> receipt crop -> grayscale -> denoise /
contrast normalization -> Gaussian filtering -> binarize -> closing -> Tesseract OCR.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "CheckFromALDI.jpg"
DEFAULT_OUTPUT = ROOT / "GermanDetect" / "output"
# Approximate paper boundary for the supplied photo (TL, TR, BR, BL).
DEFAULT_RECEIPT_CORNERS = np.array([[108, 120], [1190, 92], [1285, 1735], [108, 1772]], dtype=np.float32)


def order_points(points: np.ndarray) -> np.ndarray:
    """Return four points in top-left, top-right, bottom-right, bottom-left order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array(
        [points[np.argmin(sums)], points[np.argmin(diffs)],
         points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32
    )


def find_receipt_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """Try to find a quadrilateral receipt outline; return None if uncertain."""
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = small.shape[0] * small.shape[1]
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            candidates.append((area, polygon.reshape(4, 2).astype(np.float32) / scale))
    if not candidates:
        return None
    return order_points(max(candidates, key=lambda item: item[0])[1])


def parse_corners(value: str) -> np.ndarray:
    """Parse --corners as 'x1,y1;x2,y2;x3,y3;x4,y4' (clockwise-ish)."""
    try:
        points = [[float(part.strip()) for part in pair.split(",")] for pair in value.split(";")]
        if len(points) != 4 or any(len(point) != 2 for point in points):
            raise ValueError
        return order_points(np.asarray(points, dtype=np.float32))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use four x,y pairs separated by semicolons, e.g. '120,80;900,80;900,1500;120,1500'"
        ) from exc


def perspective_warp(image: np.ndarray, corners: Optional[np.ndarray]) -> np.ndarray:
    if corners is None:
        return image.copy()
    tl, tr, br, bl = order_points(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 2 or height < 2:
        raise ValueError("Corner points produce an invalid output size.")
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), destination)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def crop_margin(image: np.ndarray, margin: float) -> np.ndarray:
    if not 0 <= margin < 0.2:
        raise ValueError("Crop margin must be between 0 and 0.2 (fraction of each edge).")
    h, w = image.shape[:2]
    dx, dy = int(w * margin), int(h * margin)
    cropped = image[dy:h - dy, dx:w - dx]
    if cropped.size == 0:
        raise ValueError("Crop margin removed the whole image.")
    return cropped


def preprocess(image: np.ndarray, block_size: int, c_value: int, morphology: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHE evens out illumination across the wrinkled, shadowed receipt.
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Smooth residual speckles before thresholding; keep the kernel small to preserve strokes.
    smoothed = cv2.GaussianBlur(denoised, (5, 5), 0)
    binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c_value)
    result = binary
    if morphology != "none":
        # Keep this modest: a 3x3 closing bridges tiny breaks without fusing whole letters.
        operation = cv2.MORPH_OPEN if morphology == "open" else cv2.MORPH_CLOSE
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(binary, operation, kernel, iterations=1)
    return {"04_gray": gray, "05_contrast_denoised": denoised,
            "05_gaussian_filtered": smoothed, "06_binary": binary}, result


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess a receipt photo and optionally recognize it with Tesseract.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input receipt image (default: CheckFromALDI.jpg).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for intermediate and final images.")
    parser.add_argument("--corners", type=parse_corners, help="Manual corners in source-image pixels: TL;TR;BR;BL, each as x,y.")
    parser.add_argument("--crop-margin", type=float, default=0.0, help="Trim this fraction from each edge after perspective correction (0 to 0.2).")
    parser.add_argument("--block-size", type=int, default=41, help="Adaptive threshold neighborhood; odd integer >= 3.")
    parser.add_argument("--c-value", type=int, default=20, help="Adaptive threshold offset; higher values suppress faint background print-through.")
    parser.add_argument("--morphology", choices=("none", "open", "close"), default="close", help="Morphology after binarization; default close applies a gentle 3x3 closing before OCR.")
    parser.add_argument("--tesseract", type=Path, help="Tesseract executable path (auto-detected by default).")
    parser.add_argument("--lang", default="deu+eng", help="Tesseract language(s), default: deu+eng.")
    parser.add_argument("--psm", type=int, default=4, help="Tesseract page segmentation mode, default: 4 (single column).")
    parser.add_argument("--ocr-source", choices=("contrast", "ready"), default="ready",
                        help="Image used for OCR; default ready includes Gaussian smoothing, thresholding, and morphology closing.")
    parser.add_argument("--no-ocr", action="store_true", help="Only run image preprocessing; do not invoke Tesseract.")
    args = parser.parse_args()

    if args.block_size < 3 or args.block_size % 2 == 0:
        parser.error("--block-size must be an odd integer of at least 3")
    source = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if source is None:
        parser.error(f"Cannot read input image: {args.input}")

    corners = args.corners if args.corners is not None else find_receipt_corners(source)
    if corners is None:
        is_supplied_photo = args.input.resolve() == DEFAULT_INPUT.resolve() and source.shape[:2] == (1793, 1290)
        if is_supplied_photo:
            corners = DEFAULT_RECEIPT_CORNERS
            print("Automatic outline detection was uncertain; using the configured boundary for CheckFromALDI.jpg. Use --corners to override it.")
        else:
            print("Automatic outline detection was uncertain; using the full image. Use --corners to set four points manually.")
    corrected = perspective_warp(source, corners)
    cropped = crop_margin(corrected, args.crop_margin)
    stages, ocr_ready = preprocess(cropped, args.block_size, args.c_value, args.morphology)

    args.output.mkdir(parents=True, exist_ok=True)
    save_image(args.output / "01_original.jpg", source)
    save_image(args.output / "02_perspective.jpg", corrected)
    save_image(args.output / "03_receipt_crop.jpg", cropped)
    for name, stage in stages.items():
        save_image(args.output / f"{name}.png", stage)
    save_image(args.output / "07_ocr_ready.png", ocr_ready)
    print(f"Done. OCR-ready image: {args.output / '07_ocr_ready.png'}")
    if not args.no_ocr:
        executable = args.tesseract or (Path(shutil.which("tesseract")) if shutil.which("tesseract") else None)
        if executable is None:
            common_path = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe"
            if common_path.is_file():
                executable = common_path
        if executable is None or not executable.is_file():
            parser.error("Tesseract was not found. Install it or pass --tesseract <path>; use --no-ocr to skip recognition.")
        ocr_image = args.output / ("05_contrast_denoised.png" if args.ocr_source == "contrast" else "07_ocr_ready.png")
        command = [str(executable), str(ocr_image), "stdout",
                   "-l", args.lang, "--oem", "1", "--psm", str(args.psm)]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "Tesseract failed.").strip()
            parser.error(detail)
        text_path = args.output / "08_ocr_result.txt"
        text_path.write_text(result.stdout.strip() + "\n", encoding="utf-8")
        print(f"OCR text saved to: {text_path} (source: {ocr_image.name})")
    else:
        print("Tesseract OCR skipped (--no-ocr).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

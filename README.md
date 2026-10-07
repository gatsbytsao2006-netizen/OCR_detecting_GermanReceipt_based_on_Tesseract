# German receipt OCR preprocessing

A small OpenCV and Tesseract project that straightens a receipt photo, reduces image noise, and produces editable OCR text.

## Features

- Detects or accepts manual receipt corners, then applies perspective correction.
- Normalizes contrast, denoises, applies a 5×5 Gaussian blur, and thresholds uneven lighting.
- Applies a 3×3 morphological closing before recognition to reduce small gaps and faint print-through.
- Runs Tesseract using German and English (`deu+eng`) and writes UTF-8 text.
- Saves intermediate images so preprocessing can be inspected and tuned.

## Requirements

- Python 3.10 or newer
- OpenCV and NumPy (install from `requirements.txt`)
- Tesseract OCR 5 with the `deu` and `eng` language data installed

Tesseract is a separate application; the Python requirements do not install it. The script searches the system `PATH` and the standard Windows install location. Use `--tesseract` if it is installed elsewhere.

## Setup in VS Code

1. Open this folder in VS Code and select a Python interpreter.
2. Create and activate a virtual environment in the integrated PowerShell terminal:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Put the receipt photo in the project root with the name `CheckFromALDI.jpg`, or pass another path with `--input`.
4. Run `GermanDetect.py` using the selected interpreter or the included **German receipt preprocessing** launch configuration.

The receipt photo and generated artifacts are intentionally excluded from Git by `.gitignore`; this keeps personal purchase data out of repository history. You can still process them locally.

## Run

```powershell
python .\GermanDetect.py
```

The output images and OCR text are written to `GermanDetect\output\`. The final OCR input is `07_ocr_ready.png`; the recognized text is `08_ocr_result.txt`.

Useful options:

```powershell
python .\GermanDetect.py --input .\receipt.jpg --lang deu+eng --psm 4
python .\GermanDetect.py --no-ocr
python .\GermanDetect.py --ocr-source contrast
python .\GermanDetect.py --corners "120,80;900,80;900,1500;120,1500"
```

The manual corner order is top-left, top-right, bottom-right, bottom-left, using pixels in the original image. Threshold and cleanup parameters can be tuned with `--block-size`, `--c-value`, `--crop-margin`, and `--morphology`.

## Processing pipeline

1. Read the source photo.
2. Find the receipt boundary or use provided corner points, then correct perspective.
3. Crop to the receipt.
4. Convert to grayscale.
5. Normalize local contrast and denoise.
6. Apply Gaussian smoothing and adaptive thresholding.
7. Apply morphological closing to the binary image.
8. Save a white-background, dark-text OCR image.
9. Run Tesseract and save UTF-8 text.

The source receipt has folds and shadows, so inspect uncertain words and prices against the intermediate images. OCR output can require correction.

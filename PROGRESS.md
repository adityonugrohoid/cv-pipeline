# Progress

**Goal:** Build a four-phase computer vision portfolio culminating in a construction blueprint analyzer.

## Phase Status

| Phase | Name | Status | PR | Key Files | Tests |
|-------|------|--------|----|-----------|-------|
| 1 | Shape Detection | Done | [#1](https://github.com/adityonugrohoid/cv-pipeline/pull/1) | detector.py, annotator.py, export.py, cli.py | 17 passing |
| 2 | OCR Pipeline | Done | [#3](https://github.com/adityonugrohoid/cv-pipeline/pull/3) | preprocess.py, ocr_engine.py, text_regions.py, table_detector.py, cli.py | 18 passing |
| 3 | YOLO Detection | Not started | — | — | — |
| 4 | Blueprint Analyzer | Not started | — | — | — |

## Phase 1 — Completed

**What was built:**
- Contour-based shape detection for rectangles, squares, circles, triangles, and polygons
- Color-segmented masks to isolate overlapping shapes by line color, with Canny edge fallback for monochrome images
- Flood-fill step so outline shapes (unfilled circles/triangles) get correct area and circularity
- Annotator that draws color-coded bounding boxes, center dots, and labels
- JSON export with per-shape properties and summary counts
- CLI: `python -m phase1_shape_detection.cli detect --input img.png --output out.png --json results.json`
- Sample test image (`assets/sample_shapes.png`) with 8 shapes, all detected correctly
- OVERVIEW.md explaining the module in plain language

**Key decisions:**
- Extended `reference/detect_rectangles.py` color segmentation approach rather than starting from scratch
- Used morphological closing + contour fill instead of raw dilation to preserve vertex counts on triangles
- Classified squares separately from rectangles using aspect ratio (0.9–1.1 range)
- Dataclass `Shape` as the structured return type used by all downstream modules

## Phase 2 — Completed

**What was built:**
- Image preprocessing: grayscale → denoise (non-local means) → deskew (minAreaRect angle detection) → Otsu thresholding
- Tesseract OCR wrapper with word-level bounding boxes and confidence scores
- Text region grouping by y-coordinate proximity with orientation detection
- Table detector: morphological line extraction → intersection finding → grid building → per-cell OCR
- CLI with PDF support: `python -m phase2_ocr_pipeline.cli extract --input doc.png --json results.json`
- Sample test image (`assets/sample_text.png`) with title, paragraphs, 4x3 table, and notes — all extracted correctly
- OVERVIEW.md explaining the module in plain language

**Key decisions:**
- Otsu thresholding over adaptive Gaussian — cleaner results on the clean sample images, and simpler
- Non-local means denoising over median blur — better at preserving text edges while removing noise
- Table detection uses morphological line isolation (horizontal kernel, vertical kernel) rather than Hough lines — more robust for grid structures
- `TextBlock` and `TextRegion` dataclasses mirror Phase 1's `Shape` pattern
- Full text from `image_to_string()` is authoritative; region grouping is for spatial analysis

**OCR accuracy on sample image:**
- Full text: 100% — all words, numbers, and phrases extracted correctly
- Table: 4 rows x 3 cols, all 12 cells match exactly (Material/Quantity/Unit headers + 3 data rows)

## What's Next

Phase 3: YOLO Object Detection — synthetic dataset generation, YOLOv8n fine-tuning, evaluation metrics, and inference CLI.

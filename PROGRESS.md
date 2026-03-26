# Progress

**Goal:** Build a four-phase computer vision portfolio culminating in a construction blueprint analyzer.

## Phase Status

| Phase | Name | Status | PR | Key Files | Tests |
|-------|------|--------|----|-----------|-------|
| 1 | Shape Detection | Done | [#1](https://github.com/adityonugrohoid/cv-pipeline/pull/1) | detector.py, annotator.py, export.py, cli.py | 17 passing |
| 2 | OCR Pipeline | Not started | — | — | — |
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

## What's Next

Phase 2: OCR Pipeline — text extraction with Tesseract, preprocessing (deskew, denoise, threshold), text region grouping, and table detection.

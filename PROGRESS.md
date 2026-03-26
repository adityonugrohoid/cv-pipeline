# Progress

**Goal:** Build a four-phase computer vision portfolio culminating in a construction blueprint analyzer.

## Phase Status

| Phase | Name | Status | PR | Key Files | Tests |
|-------|------|--------|----|-----------|-------|
| 1 | Shape Detection | Done | [#1](https://github.com/adityonugrohoid/cv-pipeline/pull/1) | detector.py, annotator.py, export.py, cli.py | 17 passing |
| 2 | OCR Pipeline | Done | [#3](https://github.com/adityonugrohoid/cv-pipeline/pull/3) | preprocess.py, ocr_engine.py, text_regions.py, table_detector.py, cli.py | 18 passing |
| 3 | YOLO Detection | Done | [#4](https://github.com/adityonugrohoid/cv-pipeline/pull/4) | dataset.py, train.py, evaluate.py, detect.py, visualize.py, cli.py | 13 passing |
| 4 | Blueprint Analyzer | Done | [#5](https://github.com/adityonugrohoid/cv-pipeline/pull/5) | pdf_handler.py, shape_layer.py, text_layer.py, symbol_layer.py, pipeline.py, report.py, serve.py, cli.py | 18 passing |

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

## Phase 3 — Completed

**What was built:**
- Synthetic dataset generator: 200 images with 5 construction symbol classes (arrow, dimension_line, circle_x, door_swing, electrical_outlet), 80/20 train/val split, YOLO-format labels, data.yaml
- YOLOv8n fine-tuning: 20 epochs on NVIDIA RTX 4060, completed in ~1.5 minutes
- Evaluation: mAP@50, mAP@50-95, per-class precision/recall, confusion matrix
- Inference with NMS and confidence thresholding, returning `Detection` dataclass
- Visualization with color-coded class labels and confidence scores
- CLI: `python -m phase3_yolo_detection.cli {generate,train,evaluate,detect}`
- OVERVIEW.md explaining the module in plain language

**Key decisions:**
- 5 construction-specific symbol classes (arrows, dimension lines, circle-X, door swings, electrical outlets) rather than generic shapes
- YOLOv8n (nano) for speed — converges in 20 epochs on synthetic data
- Used `model.trainer.save_dir` to find weights reliably (ultralytics nests output paths)
- Synthetic dataset with slightly noisy backgrounds and randomized symbol placement for variety
- Tests that require trained weights use `pytest.skip()` when weights are unavailable

**Model performance:**
- mAP@50: 0.992, mAP@50-95: 0.898
- All 5 classes: precision > 0.98, recall > 0.95
- Best class: circle_x (AP50=0.995, mAP50-95=0.984)
- Hardest class: arrow (AP50=0.994, mAP50-95=0.828) — expected, arrows vary most in orientation

## Phase 4 — Completed

**What was built:**
- PDF handler: pdf2image conversion with configurable DPI
- Three pipeline layers wrapping Phase 1 (shapes), Phase 2 (text/OCR), Phase 3 (YOLO symbols)
- Pipeline orchestrator with per-stage timing, structured logging, and graceful failure handling
- Report generator: page-level detail + summary aggregation (shapes by type, text blocks, tables, symbols by class)
- FastAPI server: POST /analyze (upload PDF → JSON report), GET /health, Swagger UI at /docs
- CLI: `python -m phase4_blueprint_analyzer.cli analyze --input blueprint.pdf --output report.json`
- Sample blueprint PDF with room outlines, dimension lines, door swings, electrical outlets, text labels, and material table
- OVERVIEW.md explaining the module in plain language
- Root README.md covering all 4 phases

**Key decisions:**
- Each stage runs independently — errors are caught and logged, not propagated
- Symbol layer returns empty list (not error) when YOLO weights missing
- Timing per stage for observability
- Report structure: pages[] with shapes/text/symbols per page, plus summary with totals
- FastAPI uses temp file for PDF upload, cleans up after processing

**Pipeline results on sample blueprint:**
- Shapes: 87 detected (8 rectangles, 4 circles, 74 polygons, 1 triangle)
- Text: 56 blocks, 15 regions, 1 table
- Symbols: 15 YOLO detections (10 dimension_line, 2 door_swing, 2 electrical_outlet, 1 arrow)
- Timing: shapes 0.25s, text 3.74s, symbols 5.53s

## Project Complete

All four phases built, tested (66 tests passing), documented, and merged. The pipeline decomposes blueprint understanding into specialized sub-tasks — geometry, text, symbols — composed by a multi-stage orchestrator, mirroring production CV systems for technical document analysis.

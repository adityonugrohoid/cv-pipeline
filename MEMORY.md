# Memory — cv-pipeline

## Project

- **Repo:** ~/projects/cv-pipeline
- **Remote:** https://github.com/adityonugrohoid/cv-pipeline
- **Purpose:** Four-phase CV portfolio targeting a construction/logistics AI company that uses PyTorch, YOLO, OpenCV, and Tesseract for blueprint analysis
- **Build sequence:** Phase 1 → 2 → 3 → 4 — always follow CLAUDE.md build order

## Reference Files

- `reference/detect_rectangles.py` — Original rectangle detection mock test from the interview; served as the starting point for Phase 1
- `reference/sample_test.jpg`, `reference/output.jpg` — Original test artifacts from the mock test

## Phase 1: Shape Detection — Completed

- **PR:** [#1](https://github.com/adityonugrohoid/cv-pipeline/pull/1), merged to main
- **Modules:** detector.py, annotator.py, export.py, cli.py, __main__.py
- **Tests:** 17 passing (test_detector.py, test_export.py)
- **Sample image:** `assets/sample_shapes.png` — 8 shapes (2 triangles, 2 circles, 2 squares, 1 rectangle, 1 pentagon)
- **Core dataclass:** `Shape` (shape_type, contour, bbox, center, properties, confidence)
- **Detection approach:** Color segmentation → per-color binary mask → contour tracing → vertex counting + circularity. Canny edge fallback for monochrome images.
- **Key fix during build:** Morphological closing + contour flood-fill instead of raw dilation — raw dilation was rounding triangle tips (adding a 4th vertex) and leaving circle rings unfilled (breaking circularity calculation)

## Phase 2: OCR Pipeline — Completed

- **PR:** [#3](https://github.com/adityonugrohoid/cv-pipeline/pull/3), merged to main
- **Modules:** preprocess.py, ocr_engine.py, text_regions.py, table_detector.py, cli.py, __main__.py
- **Tests:** 18 passing (test_preprocess.py, test_ocr_engine.py)
- **Sample image:** `assets/sample_text.png` — title, 3 paragraphs, 4x3 table, 2 notes lines
- **Core dataclasses:** `TextBlock` (text, bbox, confidence), `TextRegion` (text, bbox, blocks, orientation), `Table` (bbox, rows, cols, cells)
- **Pipeline flow:** grayscale → denoise → deskew → Otsu threshold → Tesseract OCR → region grouping → table detection
- **System deps:** tesseract-ocr, poppler-utils (for PDF support)
- **OCR accuracy:** 100% on sample image (all text, all 12 table cells exact match)

## Phase 3: YOLO Object Detection — Completed

- **PR:** [#4](https://github.com/adityonugrohoid/cv-pipeline/pull/4), merged to main
- **Modules:** dataset.py, train.py, evaluate.py, detect.py, visualize.py, cli.py, __main__.py
- **Tests:** 13 passing (test_dataset.py, test_detect.py)
- **Trained weights:** `models/best.pt` (6.2MB, YOLOv8n fine-tuned)
- **Classes:** arrow, dimension_line, circle_x, door_swing, electrical_outlet
- **Core dataclass:** `Detection` (class_name, confidence, bbox, center)
- **Training:** 200 synthetic images (160 train / 40 val), 20 epochs, RTX 4060
- **Performance:** mAP@50=0.992, mAP@50-95=0.898, all classes P>0.98 R>0.95
- **Key fix during build:** ultralytics nests `save_dir` unpredictably — used `model.trainer.save_dir` instead of hardcoded `project/name` path to find weights

## Phase 4: Blueprint Analyzer (Capstone) — Completed

- **PR:** [#5](https://github.com/adityonugrohoid/cv-pipeline/pull/5), merged to main
- **Modules:** pdf_handler.py, shape_layer.py, text_layer.py, symbol_layer.py, pipeline.py, report.py, serve.py, cli.py, __main__.py
- **Tests:** 18 passing (test_pipeline.py, test_report.py)
- **Sample PDF:** `assets/sample_blueprint.pdf` — floor plan with rooms, dimension lines, doors, outlets, text labels, material table
- **Pipeline output:** 87 shapes, 56 text blocks, 1 table, 15 YOLO symbols on sample blueprint
- **API:** FastAPI at :8000 — POST /analyze (upload PDF), GET /health
- **Key design:** Each stage runs independently with graceful failure — symbol layer returns [] if weights missing, errors captured in report

## Conventions

- Type hints on all functions
- Google-style docstrings
- `logging` module, never `print`, in library code
- Dataclasses for structured returns (Shape, TextBlock, TextRegion, Table, Detection)
- Each phase gets an OVERVIEW.md explaining it in plain layman terms
- Each phase is independently runnable via its CLI (`python -m phaseN_*.cli`)

## Rules

- After completing each phase: update PROGRESS.md, update MEMORY.md, generate OVERVIEW.md in that phase's directory
- The sample test image is a deliberate best-case scenario (white background, distinct colors, no overlap) — not a claim about real-world performance

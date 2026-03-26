# CLAUDE.md

## Project Overview

This is `cv-pipeline`, a progressive computer vision portfolio built across four phases. Each phase builds on the previous one, culminating in a construction drawing analyzer. The project demonstrates end-to-end CV engineering capability: classical image processing, OCR, object detection with YOLO, and multi-stage document understanding pipelines.

**Target audience:** Technical interviewers at a construction/logistics AI company that uses PyTorch, YOLO, OpenCV, and Tesseract for blueprint analysis. The company decomposes blueprint understanding into multiple sub-tasks (OCR, edge detection, vector detection, semantic comprehension) rather than running a single monolithic model.

## Project Structure

```
cv-pipeline/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── phase1_shape_detection/
│   ├── __init__.py
│   ├── detector.py            # Contour-based shape detection (rect, circle, line, polygon)
│   ├── annotator.py           # Draw bounding boxes, centers, labels on image
│   ├── export.py              # Output structured JSON with coordinates, dimensions, area
│   ├── cli.py                 # CLI entrypoint: input image → annotated output + JSON
│   └── tests/
│       ├── __init__.py
│       ├── test_detector.py
│       └── test_export.py
├── phase2_ocr_pipeline/
│   ├── __init__.py
│   ├── preprocess.py          # Deskew, denoise, threshold, adaptive binarization
│   ├── ocr_engine.py          # Tesseract wrapper with confidence scores
│   ├── text_regions.py        # Detect text regions, return bounding boxes + text
│   ├── table_detector.py      # Simple table/grid detection from horizontal+vertical lines
│   ├── cli.py                 # CLI: input PDF/image → structured text output + JSON
│   └── tests/
│       ├── __init__.py
│       ├── test_preprocess.py
│       └── test_ocr_engine.py
├── phase3_yolo_detection/
│   ├── __init__.py
│   ├── dataset.py             # Dataset preparation: YOLO format, train/val split
│   ├── train.py               # Fine-tune YOLOv8 on custom dataset
│   ├── evaluate.py            # mAP, precision, recall, confusion matrix
│   ├── detect.py              # Run inference, return detections with confidence
│   ├── visualize.py           # Draw detections on image with labels and scores
│   ├── cli.py                 # CLI: train, evaluate, or detect
│   └── tests/
│       ├── __init__.py
│       ├── test_dataset.py
│       └── test_detect.py
├── phase4_blueprint_analyzer/
│   ├── __init__.py
│   ├── pipeline.py            # Multi-stage orchestrator: preprocess → OCR → detect → analyze
│   ├── pdf_handler.py         # PDF to image conversion (per-page)
│   ├── shape_layer.py         # Phase 1 shape detection as pipeline stage
│   ├── text_layer.py          # Phase 2 OCR as pipeline stage
│   ├── symbol_layer.py        # Phase 3 YOLO detection as pipeline stage
│   ├── report.py              # Generate structured takeoff report (JSON + summary)
│   ├── serve.py               # FastAPI endpoint: upload PDF → get analysis
│   ├── cli.py                 # CLI: input blueprint PDF → takeoff report
│   └── tests/
│       ├── __init__.py
│       ├── test_pipeline.py
│       └── test_report.py
├── assets/
│   ├── sample_shapes.png      # Test image for phase 1 (generated)
│   ├── sample_text.png        # Test image for phase 2 (generated)
│   ├── sample_blueprint.pdf   # Test blueprint for phase 4 (generated)
│   └── .gitkeep
├── models/                    # Saved model weights
│   └── .gitkeep
├── outputs/                   # Generated outputs (annotated images, JSON, reports)
│   └── .gitkeep
└── Dockerfile
```

## Build Sequence — Follow This Order

### Phase 1: Shape Detection (OpenCV)
Build contour-based detection for geometric shapes in images.

1. `phase1_shape_detection/detector.py`
   - Load image, convert to grayscale, apply Gaussian blur
   - Canny edge detection → findContours
   - Classify shapes by vertex count: triangle (3), rectangle (4), pentagon (5+), circle (no vertices, use circularity ratio)
   - For rectangles: compute center, width, height, area, aspect ratio
   - For circles: compute center, radius, area
   - Return list of Shape dataclasses with type, contour, bbox, center, properties

2. `phase1_shape_detection/annotator.py`
   - Draw bounding boxes (color-coded by shape type)
   - Mark centers with dots
   - Add labels: shape type + dimensions or coordinates
   - Save annotated image

3. `phase1_shape_detection/export.py`
   - Convert detection results to structured JSON
   - Include: shape_type, center_x, center_y, width, height, area, confidence
   - Summary: total shapes, count per type

4. `phase1_shape_detection/cli.py`
   - `python -m phase1_shape_detection.cli detect --input image.png --output output.png --json results.json`

5. Generate `assets/sample_shapes.png` programmatically (use OpenCV to draw rectangles, circles, triangles on white background)

6. Tests: verify detection count, shape classification, JSON schema

### Phase 2: OCR Pipeline (Tesseract + OpenCV)
Build document text extraction with preprocessing.

1. `phase2_ocr_pipeline/preprocess.py`
   - Grayscale conversion
   - Noise removal (median blur or non-local means denoising)
   - Adaptive thresholding (Otsu or adaptive Gaussian)
   - Deskew: detect skew angle via minAreaRect on contours, rotate to correct
   - Return preprocessed image

2. `phase2_ocr_pipeline/ocr_engine.py`
   - Wrapper around pytesseract
   - image_to_data() for word-level bounding boxes + confidence
   - image_to_string() for full text
   - Filter low-confidence results (configurable threshold, default 60)
   - Return structured results: list of TextBlock(text, bbox, confidence)

3. `phase2_ocr_pipeline/text_regions.py`
   - Group nearby text blocks into regions (cluster by y-coordinate proximity)
   - Detect text orientation (horizontal vs vertical)
   - Return TextRegion objects with bounding box and concatenated text

4. `phase2_ocr_pipeline/table_detector.py`
   - Detect horizontal and vertical lines using morphological operations
   - Find intersections to identify grid structure
   - Extract cell contents via OCR within each cell bbox
   - Return Table object with rows and columns

5. `phase2_ocr_pipeline/cli.py`
   - `python -m phase2_ocr_pipeline.cli extract --input document.png --json results.json`
   - Support PDF input via pdf2image conversion

6. Generate `assets/sample_text.png` programmatically (use PIL to render text with known content for testing)

7. Tests: verify text extraction accuracy on known sample, preprocessing output dimensions

### Phase 3: YOLO Object Detection (PyTorch + ultralytics)
Fine-tune YOLOv8 on a small custom dataset.

1. `phase3_yolo_detection/dataset.py`
   - Generate a small synthetic dataset (100-200 images) of simple shapes/symbols
   - Use OpenCV to draw symbols (arrows, circles with X, squares with text, dimension lines)
   - Auto-generate YOLO format labels: class_id center_x center_y width height (normalized)
   - Split into train/val (80/20)
   - Create data.yaml for ultralytics

2. `phase3_yolo_detection/train.py`
   - Load YOLOv8n (nano, fastest) pretrained on COCO
   - Fine-tune on custom dataset
   - Max 20 epochs (demo speed, should converge fast on synthetic data)
   - Save best weights to models/

3. `phase3_yolo_detection/evaluate.py`
   - Load trained model, run validation
   - Report: mAP@50, mAP@50-95, per-class precision/recall
   - Generate confusion matrix visualization

4. `phase3_yolo_detection/detect.py`
   - Load model, run inference on single image
   - Return list of Detection(class_name, confidence, bbox, center)
   - Apply NMS and confidence thresholding

5. `phase3_yolo_detection/visualize.py`
   - Draw detections on image with class labels and confidence scores
   - Color-coded by class

6. `phase3_yolo_detection/cli.py`
   - `python -m phase3_yolo_detection.cli train --data data.yaml --epochs 20`
   - `python -m phase3_yolo_detection.cli detect --input image.png --output output.png --weights models/best.pt`

7. Tests: verify dataset generation, detection output format, model loading

### Phase 4: Blueprint Analyzer (Capstone)
Multi-stage pipeline combining all three phases.

1. `phase4_blueprint_analyzer/pdf_handler.py`
   - Convert PDF pages to images using pdf2image
   - Return list of page images with page numbers

2. `phase4_blueprint_analyzer/shape_layer.py`
   - Wrap phase 1 detector as a pipeline stage
   - Input: image → Output: list of detected shapes with metadata

3. `phase4_blueprint_analyzer/text_layer.py`
   - Wrap phase 2 OCR as a pipeline stage
   - Input: image → Output: list of text regions with content

4. `phase4_blueprint_analyzer/symbol_layer.py`
   - Wrap phase 3 YOLO detector as a pipeline stage
   - Input: image → Output: list of detected symbols with class and location

5. `phase4_blueprint_analyzer/pipeline.py`
   - Orchestrator that runs all stages in sequence
   - Input: PDF file path
   - Steps: pdf_to_images → for each page: shape_layer + text_layer + symbol_layer → merge results
   - Each stage runs independently (can fail gracefully without blocking others)
   - Structured logging at each stage for observability

6. `phase4_blueprint_analyzer/report.py`
   - Aggregate results into a takeoff report
   - JSON output: page-by-page with shapes, text, symbols
   - Summary: total shapes by type, extracted text blocks, detected symbols
   - Optionally render an annotated composite image

7. `phase4_blueprint_analyzer/serve.py`
   - FastAPI app
   - POST /analyze: upload PDF → run pipeline → return JSON report
   - GET /health
   - Swagger UI at /docs

8. Generate `assets/sample_blueprint.pdf` programmatically (use reportlab or PIL to create a simple floor plan with shapes, text labels, and dimension lines)

9. Tests: verify pipeline orchestration, report schema, graceful failure handling

## Requirements

```
# Core CV
opencv-python
numpy
Pillow

# OCR
pytesseract
pdf2image

# YOLO / Deep Learning
ultralytics
torch
torchvision

# Serving
fastapi
uvicorn
python-multipart

# Report generation
reportlab

# Testing
pytest

# Utilities
matplotlib
```

System dependencies (install separately):
- tesseract-ocr: `sudo apt-get install tesseract-ocr`
- poppler-utils: `sudo apt-get install poppler-utils` (for pdf2image)

## Code Standards

- Type hints on all functions
- Google-style docstrings
- Logging module (no print in library code)
- Dataclasses for structured returns (Shape, TextBlock, Detection, etc.)
- Each phase is independently runnable via its CLI
- Phase 4 imports from phases 1-3 as internal modules

## Commands

```bash
# Install
pip install -r requirements.txt
sudo apt-get install tesseract-ocr poppler-utils

# Phase 1: Shape detection
python -m phase1_shape_detection.cli detect --input assets/sample_shapes.png --output outputs/shapes_detected.png --json outputs/shapes.json

# Phase 2: OCR
python -m phase2_ocr_pipeline.cli extract --input assets/sample_text.png --json outputs/text.json

# Phase 3: YOLO training and detection
python -m phase3_yolo_detection.cli train --epochs 20
python -m phase3_yolo_detection.cli detect --input assets/sample_shapes.png --output outputs/yolo_detected.png

# Phase 4: Full pipeline
python -m phase4_blueprint_analyzer.cli analyze --input assets/sample_blueprint.pdf --output outputs/report.json

# API server
python -m phase4_blueprint_analyzer.serve

# Tests
pytest -v

# Docker
docker build -t cv-pipeline .
docker run -p 8000:8000 cv-pipeline
```

## Key Design Principle

This pipeline decomposes document understanding into multiple specialized sub-tasks rather than running a single monolithic model. Each phase handles one aspect (geometry, text, symbols) and the capstone orchestrator merges results. This mirrors how production CV systems for technical documents actually work: you don't ask one model to do everything, you compose multiple specialized components.

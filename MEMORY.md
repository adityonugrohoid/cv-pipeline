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

## Conventions

- Type hints on all functions
- Google-style docstrings
- `logging` module, never `print`, in library code
- Dataclasses for structured returns (Shape, and future: TextBlock, Detection)
- Each phase gets an OVERVIEW.md explaining it in plain layman terms
- Each phase is independently runnable via its CLI (`python -m phaseN_*.cli`)

## Rules

- After completing each phase: update PROGRESS.md, update MEMORY.md, generate OVERVIEW.md in that phase's directory
- The sample test image is a deliberate best-case scenario (white background, distinct colors, no overlap) — not a claim about real-world performance

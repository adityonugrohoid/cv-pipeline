# CV Pipeline — Concept & Boon AI Mapping

## Purpose

A progressive CV portfolio that maps directly to Boon AI's confirmed tech stack and blueprint analysis pipeline. Built in 4 phases, each independently demonstrable, culminating in a multi-stage blueprint analyzer that mirrors Boon's production architecture.

## Why This Matters for Boon

Boon's founder said: "Most AI vision systems treat construction drawings as a single detection problem. That approach breaks down quickly. Blueprints aren't just images." Boon decomposes blueprint analysis into multiple sub-tasks. This repo demonstrates exactly that approach.

## Boon Stack → Portfolio Mapping

| Boon's Confirmed Stack | Portfolio Phase | What We Demonstrate |
|---|---|---|
| OpenCV | Phase 1 | Contour detection, shape classification, geometric analysis |
| Tesseract | Phase 2 | OCR with preprocessing, text region grouping, table detection |
| YOLO | Phase 3 | Fine-tuning YOLOv8 on custom dataset, inference pipeline |
| PyTorch | Phase 3 | Model training, evaluation, metrics |
| Multi-stage pipeline | Phase 4 | Orchestrator composing shape + text + symbol layers |
| FastAPI serving | Phase 4 | Upload PDF → get structured analysis |
| Document AI | Phase 4 | PDF handling, page-level processing, structured reports |

## Boon Job Posting Requirements → Evidence

| Requirement (from Applied ML Engineer CV posting) | Evidence in This Repo |
|---|---|
| "Build and deploy models for OCR, edge detection, vector detection" | Phases 1, 2, 3 |
| "Semantic comprehension of construction drawings" | Phase 4 pipeline |
| "Own components from data annotation tools to model training scripts to inference services" | Full pipeline ownership across all phases |
| "Fine-tune state-of-the-art CV models" | Phase 3 YOLO fine-tuning |
| "Adapt open-source solutions or build custom architectures" | YOLO adaptation + custom OpenCV pipeline |
| "PyTorch, TensorFlow, OpenCV, Tesseract, YOLO" | All used across phases |

## Build Timeline (Recommended)

| Phase | Estimated Time | Priority |
|---|---|---|
| Phase 1: Shape Detection | 2-3 hours | Do first — extends today's mock test |
| Phase 2: OCR Pipeline | 3-4 hours | High — Tesseract is in Boon's stack |
| Phase 3: YOLO Detection | 4-5 hours | High — YOLO is explicitly listed |
| Phase 4: Blueprint Analyzer | 3-4 hours | Capstone — ties everything together |

Total: ~12-16 hours of Claude Code development time.

## Talking Points for Boon Technical Interview

Phase 1: "I built a contour-based shape detector that classifies rectangles, circles, triangles, and polygons from construction-style drawings, outputs structured JSON with coordinates and dimensions."

Phase 2: "The OCR pipeline preprocesses scanned documents with deskew, denoising, and adaptive thresholding before running Tesseract. It groups text into regions and can detect table structures from line intersections."

Phase 3: "I fine-tuned YOLOv8 on a synthetic dataset of construction symbols. The key learning was generating quality training data, because in construction AI you often don't have labeled datasets, you have to create them."

Phase 4: "The capstone is a multi-stage pipeline that decomposes blueprint analysis into shape detection, text extraction, and symbol recognition, then merges the results into a structured takeoff report. Each stage runs independently with graceful failure, same architecture principle your team uses."

## What This Repo Does NOT Cover (mention as awareness)

- Scale: production systems process thousands of pages; this is a single-blueprint demo
- Domain-specific training data: real construction symbol datasets are proprietary
- Accuracy benchmarks: Boon claims 85%+ on mechanical ductwork; our synthetic data won't match
- Integration: Boon connects to TMS/ERP/WMS; this is standalone

Be honest about these gaps. The repo shows engineering capability and architectural understanding, not domain expertise.

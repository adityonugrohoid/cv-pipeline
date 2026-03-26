# Phase 4: Blueprint Analyzer — How It Works

## What Problem Does This Solve?

The first three phases each do one thing well: Phase 1 finds shapes, Phase 2 reads text, Phase 3 recognizes construction symbols. But a real blueprint has all of these at once — room outlines, dimension labels, door swings, material tables — and you need one system that handles the whole document. Phase 4 is the conductor of the orchestra. It takes a PDF blueprint, converts each page to an image, runs all three detection stages on it, and produces a single structured report covering everything found. If one stage fails (say the YOLO model is missing), the other stages keep running.

## Why This Architecture Matters

This project deliberately avoids using a single monolithic model. Instead, it decomposes document understanding into specialized sub-tasks — geometry, text, and symbols — and composes them together. This mirrors how production systems at construction-tech companies actually work. A monolithic model would need to be retrained for every new task. With a pipeline, you can swap out just the symbol detector, or upgrade the OCR engine, without touching the rest.

## How It Works, Step by Step

1. **You give it a PDF.** The system accepts a blueprint document (one or more pages).

2. **It converts pages to images.** Each PDF page is rendered into a high-resolution image. This is necessary because the detection stages work on images, not on PDF vector data directly.

3. **It runs three detection stages per page.** For each page image, the pipeline runs — in sequence — shape detection (Phase 1), text extraction (Phase 2), and symbol recognition (Phase 3). Each stage runs independently. If one crashes or the YOLO weights are missing, the other two still complete and their results are preserved.

4. **It aggregates results.** All per-page results are merged into a single report. The report includes page-level detail (what was found on each page) and a summary (total shapes by type, total text blocks, total symbols by class).

5. **It exports a JSON report.** The final output is a structured data file that another program (or a human) can read, search, or feed into a downstream system.

6. **Optionally, it serves over HTTP.** A FastAPI web server exposes a `POST /analyze` endpoint. You upload a PDF, and it returns the JSON report — no command line needed.

## What Each File Does

**pdf_handler.py** — The translator. PDFs are a document format, not an image format. This file converts each page of a PDF into a pixel image that the detection stages can process. Think of it as photocopying each page of a booklet into individual photos.

**shape_layer.py** — The shape spotter. A thin wrapper around Phase 1's detector. It takes an image, runs contour-based shape detection, and returns the results as a simple list of dictionaries. It translates Phase 1's internal data structures into the format the pipeline expects.

**text_layer.py** — The reader. A thin wrapper around Phase 2's OCR pipeline. It preprocesses the image (denoise, deskew, threshold), runs Tesseract for text extraction, groups text into regions, and detects tables. Returns everything as a dictionary.

**symbol_layer.py** — The symbol recognizer. A thin wrapper around Phase 3's YOLO detector. It runs the trained model on the image and returns detected construction symbols. If the model weights file is missing, it gracefully returns an empty list instead of crashing.

**pipeline.py** — The conductor. This is the orchestrator. For each page, it calls shape_layer, text_layer, and symbol_layer in sequence, catching any errors so one failure does not block the others. It logs timing for each stage (how long shapes took, how long OCR took, etc.) for observability. It supports both single-image analysis and full PDF processing.

**report.py** — The summarizer. Takes the raw per-page results from the pipeline and aggregates them into a structured report. It counts total shapes by type, total text blocks and tables, and total symbols by class. It also collects any errors that occurred during processing.

**serve.py** — The web server. A FastAPI application with two endpoints: `GET /health` (is the server running?) and `POST /analyze` (upload a PDF, get back a JSON report). The Swagger UI at `/docs` provides interactive documentation.

**cli.py** — The front door. Command-line interface with two modes: `analyze` (run the full pipeline on a PDF) and `generate` (create a sample blueprint PDF for testing).

## What the Output Looks Like

The JSON report has this structure (abbreviated):

```
{
  "pages": [
    {
      "page": 1,
      "shapes": [ {"shape_type": "rectangle", "bbox": [...], ...}, ... ],
      "text": {
        "full_text": "Floor Plan — Building A\nOffice 101\n25'-0\" x 20'-0\"...",
        "text_blocks": 56,
        "text_regions": [ {"text": "...", "bbox": [...]} ],
        "tables": [ {"rows": 4, "cols": 3, "cells": [["Material", "Qty", "Unit"], ...]} ]
      },
      "symbols": [ {"class": "door_swing", "confidence": 0.94, "bbox": [...]} ],
      "timing": {"shapes_sec": 0.25, "text_sec": 3.74, "symbols_sec": 5.53},
      "errors": {}
    }
  ],
  "summary": {
    "total_pages": 1,
    "shapes": {"total": 87, "by_type": {"rectangle": 8, "circle": 4, ...}},
    "text": {"total_blocks": 56, "total_tables": 1},
    "symbols": {"total": 15, "by_class": {"door_swing": 2, "dimension_line": 10, ...}}
  }
}
```

## Key Concepts Explained

**Pipeline architecture** — Instead of one model doing everything, you chain multiple specialized tools in sequence. Each tool handles one aspect of the problem. This is like an assembly line in a factory — one station cuts, another welds, another paints — rather than having one person do everything.

**Graceful failure** — If one stage crashes (say Tesseract is not installed, or the YOLO weights are missing), the pipeline catches the error and continues with the other stages. You get partial results instead of nothing. This is critical in production systems where reliability matters more than perfection.

**Structured logging** — Each stage logs when it starts, how long it takes, and whether it succeeded or failed. This makes it possible to diagnose problems after the fact — like a flight recorder for your software.

**Takeoff report** — In construction, a "takeoff" is the process of measuring and listing all the materials, components, and quantities from a set of blueprints. The report this pipeline generates is a simplified version of that: what shapes exist, what text says, what symbols appear.

**FastAPI** — A modern web framework that automatically generates interactive API documentation (Swagger UI at `/docs`). It lets you upload a PDF through a web browser or any HTTP client and get back the analysis as JSON.

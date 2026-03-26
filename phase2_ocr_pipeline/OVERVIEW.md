# Phase 2: OCR Pipeline — How It Works

## What Problem Does This Solve?

Phase 1 can find shapes in an image, but it cannot read. If a construction drawing says "Building A — 2500 sq ft" next to a rectangle, Phase 1 sees the rectangle but ignores the words entirely. Phase 2 gives the system the ability to read text from images. It takes a picture of a document — a scanned page, a photo of a blueprint label, a PDF — and pulls out the words, including where each word sits on the page and how confident it is about the reading. It can also detect tables (grids of rows and columns) and extract the contents of each cell.

## Why the Sample Image Is Intentionally Easy

Just like Phase 1's shape image, the test image (`assets/sample_text.png`) is a best-case scenario: clean white background, crisp computer-rendered fonts, no wrinkles or shadows, perfectly straight text. Real scanned documents have smudges, skewed pages, faded ink, and background noise. The preprocessing steps (denoising, deskewing, thresholding) exist to bridge that gap, but they cannot fix everything. This sample proves the pipeline works correctly when given clean input — handling messy real-world scans is a harder problem that benefits from the combined pipeline in Phase 4.

## How It Works, Step by Step

1. **You give it a picture or PDF.** The system accepts image files (PNG, JPG) or PDFs. For PDFs, it converts each page into an image first.

2. **It cleans up the image.** This is the preprocessing stage — think of it as preparing a messy handwritten note before photocopying it. The system converts the image to black-and-white, removes speckles and noise (like wiping dust off a scanner), straightens the page if it was scanned at an angle, and then sharpens the contrast so letters are crisp black on pure white.

3. **It reads every word.** The cleaned image goes to Tesseract, an open-source text recognition engine (think of it as a very fast reader that looks at each word and guesses what it says). For every word, Tesseract reports what it thinks the word is, where it sits on the page, and a confidence score — how sure it is about its guess. Low-confidence guesses (below 60% by default) get filtered out.

4. **It groups words into regions.** Individual words are not very useful on their own. The system looks at which words are near each other vertically and groups them into logical chunks — a title, a paragraph, a label. This is like looking at a page and noticing "these five words form a sentence at the top, and those ten words form a paragraph below."

5. **It finds tables.** The system looks for grids — patterns of horizontal and vertical lines that form rows and columns. It finds where lines cross to map out the grid structure, then reads the text inside each cell. This turns a visual table in an image into structured data you can work with.

6. **It packages the results.** Everything gets written to a JSON file: the full extracted text, the grouped regions with their positions, and any tables with their cell contents.

## What Each File Does

**preprocess.py** — The janitor. Before the system can read anything, the image needs cleaning. This file handles four steps: converting color images to grayscale (like photocopying in black-and-white), removing noise (smoothing out speckles and scanner artifacts), straightening skewed pages (detecting if the page is tilted and rotating it back), and thresholding (forcing every pixel to be either pure black or pure white, so letters stand out sharply). Each step makes the next one's job easier.

**ocr_engine.py** — The reader. This is the wrapper around Tesseract, the text recognition engine. It can work in two modes: word-by-word mode (returning each word with its exact position on the page and a confidence percentage) or full-text mode (returning all the text as one block, like copying and pasting from the image). It filters out low-confidence guesses so you don't get garbage mixed in with real text.

**text_regions.py** — The organizer. Raw OCR gives you a flat list of individual words scattered across the page. This file groups them into meaningful regions by looking at vertical spacing — words that are close together vertically belong to the same block of text, and a big vertical gap means a new section starts. It also checks whether text runs horizontally (normal) or vertically (like a rotated sidebar label).

**table_detector.py** — The grid reader. This file finds tables by looking for line patterns. It uses a clever trick: it searches for long horizontal lines and long vertical lines separately, then finds where they cross. Those crossing points define the corners of table cells. Once it knows the grid layout, it crops each cell and runs OCR on it individually. The result is a structured table — rows and columns of text — rather than a jumbled mess of words that happened to be inside a box.

**cli.py** — The front door. Like Phase 1, this ties everything together into a command you can run from a terminal. It also handles PDF conversion (turning PDF pages into images before processing) and includes a generator that creates the sample test image.

## What the Output Looks Like

The JSON output has this structure (abbreviated):

```
{
  "pages": [
    {
      "page": 1,
      "full_text": "Project Specifications\n\nBuilding A is located...",
      "text_regions": [
        {
          "text": "Project Specifications Building A is located...",
          "bbox": [50, 34, 444, 124],
          "orientation": "horizontal",
          "block_count": 26
        }
      ],
      "tables": [
        {
          "rows": 4,
          "cols": 3,
          "cells": [
            ["Material", "Quantity", "Unit"],
            ["Concrete", "150", "cubic yards"],
            ["Steel Rebar", "2000", "linear feet"],
            ["Lumber", "500", "board feet"]
          ]
        }
      ]
    }
  ],
  "summary": { "total_pages": 1, "total_text_blocks": 39, "total_tables": 1 }
}
```

The `full_text` gives you everything on the page as plain text. The `text_regions` tell you where each block of text sits. The `tables` give you cell-by-cell contents in row/column order — ready to drop into a spreadsheet.

## Key Concepts Explained

**OCR (Optical Character Recognition)** — The technology that converts pictures of text into actual text a computer can search, copy, and process. Think of it like a person reading a photograph of a book page and typing out what they see.

**Preprocessing** — Cleaning up an image before OCR reads it. Tesseract works best on crisp black text on a white background, so preprocessing tries to get as close to that ideal as possible. It is like washing and ironing a shirt before a photo — the subject is the same, but the result looks much better.

**Thresholding** — Forcing every pixel to be either black or white with nothing in between. Imagine turning up the contrast on a photo until grays disappear — faint pencil marks become solid black, and light shadows become pure white. This makes letters sharper and easier for OCR to read.

**Deskewing** — Straightening a tilted page. When you scan a document and it comes out slightly crooked, deskewing detects the tilt angle and rotates the image to make text lines perfectly horizontal. OCR accuracy drops significantly on slanted text, so this step matters.

**Morphological operations** — A way to manipulate shapes in a binary image by growing or shrinking white regions. Table detection uses this to isolate long horizontal lines (by stretching a filter sideways) and long vertical lines (by stretching it up and down). It is like looking at an image through a narrow slit — if you hold the slit horizontally, you only see horizontal features.

**Confidence score** — Tesseract's self-assessment of how likely it got a word right, from 0 (wild guess) to 100 (very sure). Filtering by confidence (e.g., keeping only words above 60%) removes garbled readings from noise or smudges while keeping the real text.

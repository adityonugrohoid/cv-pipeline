"""CLI entrypoint for Phase 2 OCR pipeline.

Usage:
    python -m phase2_ocr_pipeline.cli extract --input document.png --json results.json
    python -m phase2_ocr_pipeline.cli generate  # create sample test image
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2

from .ocr_engine import extract_full_text, extract_text_blocks
from .preprocess import preprocess
from .table_detector import detect_tables
from .text_regions import group_into_regions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_image(input_path: Path) -> list[tuple[int, Any]]:
    """Load an image or PDF, returning list of (page_num, image) tuples."""
    if input_path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(str(input_path))
        import numpy as np
        return [
            (i + 1, cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR))
            for i, page in enumerate(pages)
        ]
    else:
        image = cv2.imread(str(input_path))
        if image is None:
            logger.error("Failed to read image: %s", input_path)
            sys.exit(1)
        return [(1, image)]


def _page_to_dict(
    page_num: int,
    image: Any,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Run the full OCR pipeline on a single page and return results dict."""
    preprocessed = preprocess(image)

    blocks = extract_text_blocks(preprocessed, confidence_threshold=confidence_threshold)
    full_text = extract_full_text(preprocessed)
    regions = group_into_regions(blocks)
    tables = detect_tables(image, binary=preprocessed)

    return {
        "page": page_num,
        "full_text": full_text,
        "text_regions": [
            {
                "text": r.text,
                "bbox": list(r.bbox),
                "orientation": r.orientation,
                "block_count": len(r.blocks),
            }
            for r in regions
        ],
        "tables": [
            {
                "bbox": list(t.bbox),
                "rows": t.rows,
                "cols": t.cols,
                "cells": t.cells,
            }
            for t in tables
        ],
        "stats": {
            "text_blocks": len(blocks),
            "text_regions": len(regions),
            "tables": len(tables),
        },
    }


def generate_sample_image(output_path: str = "assets/sample_text.png") -> None:
    """Generate a sample image with known text for testing OCR."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 800, 600
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Title
    draw.text((50, 30), "Project Specifications", fill="black", font=font_large)

    # Paragraph text
    lines = [
        "Building A is located on the north side of the site.",
        "Total floor area is 2500 square feet.",
        "Construction start date is March 2025.",
    ]
    y = 80
    for line in lines:
        draw.text((50, y), line, fill="black", font=font_medium)
        y += 30

    # Section header
    draw.text((50, 190), "Material Schedule", fill="black", font=font_large)

    # Draw a simple table
    table_x, table_y = 50, 240
    col_widths = [200, 120, 120]
    row_height = 35
    headers = ["Material", "Quantity", "Unit"]
    rows = [
        ["Concrete", "150", "cubic yards"],
        ["Steel Rebar", "2000", "linear feet"],
        ["Lumber", "500", "board feet"],
    ]

    # Draw grid lines
    total_w = sum(col_widths)
    total_rows = 1 + len(rows)
    for r in range(total_rows + 1):
        y = table_y + r * row_height
        draw.line([(table_x, y), (table_x + total_w, y)], fill="black", width=2)
    x = table_x
    for cw in col_widths:
        for r in range(total_rows + 1):
            y_top = table_y
            y_bot = table_y + total_rows * row_height
        draw.line([(x, y_top), (x, y_bot)], fill="black", width=2)
        x += cw
    draw.line([(x, table_y), (x, table_y + total_rows * row_height)], fill="black", width=2)

    # Fill header text
    x = table_x
    for i, header in enumerate(headers):
        draw.text((x + 10, table_y + 8), header, fill="black", font=font_small)
        x += col_widths[i]

    # Fill row text
    for r, row in enumerate(rows):
        x = table_x
        for c, cell in enumerate(row):
            draw.text((x + 10, table_y + (r + 1) * row_height + 8), cell, fill="black", font=font_small)
            x += col_widths[c]

    # Footer text
    draw.text((50, 420), "Notes:", fill="black", font=font_large)
    draw.text((50, 460), "All measurements are approximate.", fill="black", font=font_medium)
    draw.text((50, 490), "Verify dimensions on site before ordering.", fill="black", font=font_medium)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    logger.info("Generated sample text image → %s", path)


def cmd_extract(args: argparse.Namespace) -> None:
    """Run OCR extraction on an input image or PDF."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    pages = _load_image(input_path)
    results: list[dict[str, Any]] = []

    for page_num, image in pages:
        logger.info("Processing page %d...", page_num)
        page_result = _page_to_dict(page_num, image, args.confidence)
        results.append(page_result)

    output = {
        "pages": results,
        "summary": {
            "total_pages": len(results),
            "total_text_blocks": sum(p["stats"]["text_blocks"] for p in results),
            "total_tables": sum(p["stats"]["tables"] for p in results),
        },
    }

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Exported results to %s", args.json)

    # Print summary
    for page in results:
        logger.info("Page %d: %d text blocks, %d regions, %d tables",
                     page["page"], page["stats"]["text_blocks"],
                     page["stats"]["text_regions"], page["stats"]["tables"])
        if page["full_text"]:
            preview = page["full_text"][:200].replace("\n", " ")
            logger.info("  Text preview: %s...", preview)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phase2_ocr_pipeline",
        description="OCR text extraction from document images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = sub.add_parser("extract", help="Extract text from image or PDF")
    p_extract.add_argument("--input", required=True, help="Input image or PDF path")
    p_extract.add_argument("--json", help="Output JSON results path")
    p_extract.add_argument("--confidence", type=float, default=60.0,
                           help="Minimum OCR confidence threshold (0-100)")

    # generate
    p_gen = sub.add_parser("generate", help="Generate sample text image")
    p_gen.add_argument("--output", default="assets/sample_text.png",
                       help="Output path for generated image")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "generate":
        generate_sample_image(args.output)


if __name__ == "__main__":
    main()

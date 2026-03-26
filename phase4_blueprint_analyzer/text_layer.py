"""Pipeline stage wrapping Phase 2 OCR extraction."""

import logging
from typing import Any

import numpy as np

from phase2_ocr_pipeline.ocr_engine import extract_full_text, extract_text_blocks
from phase2_ocr_pipeline.preprocess import preprocess
from phase2_ocr_pipeline.table_detector import detect_tables
from phase2_ocr_pipeline.text_regions import group_into_regions

logger = logging.getLogger(__name__)


def run(image: np.ndarray) -> dict[str, Any]:
    """Extract text, regions, and tables from an image.

    Args:
        image: BGR image.

    Returns:
        Dictionary with full_text, text_regions, and tables.
    """
    preprocessed = preprocess(image)

    blocks = extract_text_blocks(preprocessed)
    full_text = extract_full_text(preprocessed)
    regions = group_into_regions(blocks)
    tables = detect_tables(image, binary=preprocessed)

    logger.info("Text layer: %d blocks, %d regions, %d tables",
                len(blocks), len(regions), len(tables))

    return {
        "full_text": full_text,
        "text_blocks": len(blocks),
        "text_regions": [
            {
                "text": r.text,
                "bbox": list(r.bbox),
                "orientation": r.orientation,
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
    }

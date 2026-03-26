"""Multi-stage orchestrator that runs shape, text, and symbol detection per page.

Each stage runs independently — if one fails, the others still complete.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import shape_layer, symbol_layer, text_layer
from .pdf_handler import pdf_to_images

logger = logging.getLogger(__name__)


def _run_stage(name: str, fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float, str | None]:
    """Run a single pipeline stage with timing and error handling.

    Returns:
        Tuple of (result, elapsed_seconds, error_message_or_None).
    """
    logger.info("  [%s] Starting...", name)
    start = time.monotonic()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.monotonic() - start
        logger.info("  [%s] Completed in %.2fs", name, elapsed)
        return result, elapsed, None
    except Exception as e:
        elapsed = time.monotonic() - start
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("  [%s] Failed in %.2fs — %s", name, elapsed, error_msg)
        return None, elapsed, error_msg


def analyze_image(
    image: np.ndarray,
    page_num: int = 1,
    yolo_weights: str | Path = "models/best.pt",
) -> dict[str, Any]:
    """Run all detection stages on a single image.

    Args:
        image: BGR image.
        page_num: Page number for reporting.
        yolo_weights: Path to YOLO model weights.

    Returns:
        Dictionary with per-stage results and timing.
    """
    logger.info("Analyzing page %d (%dx%d)...", page_num, image.shape[1], image.shape[0])

    shapes, shape_time, shape_err = _run_stage("shapes", shape_layer.run, image)
    text, text_time, text_err = _run_stage("text", text_layer.run, image)
    symbols, symbol_time, symbol_err = _run_stage(
        "symbols", symbol_layer.run, image, weights=yolo_weights,
    )

    return {
        "page": page_num,
        "shapes": shapes or [],
        "text": text or {"full_text": "", "text_blocks": 0, "text_regions": [], "tables": []},
        "symbols": symbols or [],
        "timing": {
            "shapes_sec": round(shape_time, 3),
            "text_sec": round(text_time, 3),
            "symbols_sec": round(symbol_time, 3),
        },
        "errors": {
            k: v for k, v in [
                ("shapes", shape_err),
                ("text", text_err),
                ("symbols", symbol_err),
            ] if v is not None
        },
    }


def analyze_pdf(
    pdf_path: str | Path,
    yolo_weights: str | Path = "models/best.pt",
    dpi: int = 200,
) -> list[dict[str, Any]]:
    """Run the full pipeline on a PDF document.

    Args:
        pdf_path: Path to the PDF file.
        yolo_weights: Path to YOLO model weights.
        dpi: PDF rendering resolution.

    Returns:
        List of per-page result dictionaries.
    """
    pages = pdf_to_images(pdf_path, dpi=dpi)
    results: list[dict[str, Any]] = []

    for page_num, image in pages:
        page_result = analyze_image(image, page_num=page_num, yolo_weights=yolo_weights)
        results.append(page_result)

    logger.info("Pipeline complete: %d pages processed", len(results))
    return results

"""Convert PDF pages to images for pipeline processing."""

import logging
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[tuple[int, np.ndarray]]:
    """Convert each page of a PDF to a BGR numpy array.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. Higher = better quality but slower.

    Returns:
        List of (page_number, image) tuples. Page numbers are 1-based.
    """
    pdf_path = Path(pdf_path)
    logger.info("Converting PDF to images: %s (dpi=%d)", pdf_path.name, dpi)

    pil_images = convert_from_path(str(pdf_path), dpi=dpi)

    pages: list[tuple[int, np.ndarray]] = []
    for i, pil_img in enumerate(pil_images):
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        pages.append((i + 1, bgr))
        logger.info("  Page %d: %dx%d", i + 1, bgr.shape[1], bgr.shape[0])

    return pages

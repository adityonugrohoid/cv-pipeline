"""Tesseract OCR wrapper with word-level bounding boxes and confidence scores."""

import logging
from dataclasses import dataclass

import numpy as np
import pytesseract

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A detected text element with its location and confidence."""

    text: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float


def extract_text_blocks(
    image: np.ndarray,
    confidence_threshold: float = 60.0,
    lang: str = "eng",
) -> list[TextBlock]:
    """Extract word-level text blocks from an image using Tesseract.

    Args:
        image: Preprocessed image (grayscale or binary).
        confidence_threshold: Minimum confidence (0-100) to include a word.
        lang: Tesseract language code.

    Returns:
        List of TextBlock objects for words above the confidence threshold.
    """
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

    blocks: list[TextBlock] = []
    n = len(data["text"])

    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue

        conf = float(data["conf"][i])
        if conf < confidence_threshold:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        blocks.append(TextBlock(text=text, bbox=(x, y, w, h), confidence=conf))

    logger.info("Extracted %d text blocks (threshold=%.0f)", len(blocks), confidence_threshold)
    return blocks


def extract_full_text(image: np.ndarray, lang: str = "eng") -> str:
    """Extract full text from an image as a single string.

    Args:
        image: Preprocessed image (grayscale or binary).
        lang: Tesseract language code.

    Returns:
        Extracted text.
    """
    text = pytesseract.image_to_string(image, lang=lang)
    return text.strip()

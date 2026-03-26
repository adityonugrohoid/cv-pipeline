"""Tests for phase2_ocr_pipeline.ocr_engine."""

import cv2
import numpy as np
import pytest

from phase2_ocr_pipeline.ocr_engine import TextBlock, extract_full_text, extract_text_blocks
from phase2_ocr_pipeline.preprocess import preprocess


def _make_text_image(text: str = "Hello World", font_scale: float = 1.5) -> np.ndarray:
    """Create a clean image with known text for OCR testing."""
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    return img


class TestExtractTextBlocks:
    def test_returns_text_blocks(self) -> None:
        img = _make_text_image("Testing OCR")
        preprocessed = preprocess(img)
        blocks = extract_text_blocks(preprocessed)
        assert len(blocks) > 0
        assert all(isinstance(b, TextBlock) for b in blocks)

    def test_text_block_has_bbox(self) -> None:
        img = _make_text_image("Test")
        preprocessed = preprocess(img)
        blocks = extract_text_blocks(preprocessed)
        assert len(blocks) > 0
        b = blocks[0]
        assert len(b.bbox) == 4
        assert all(isinstance(v, int) for v in b.bbox)

    def test_confidence_filtering(self) -> None:
        img = _make_text_image("Clear Text", font_scale=2.0)
        preprocessed = preprocess(img)
        all_blocks = extract_text_blocks(preprocessed, confidence_threshold=0)
        high_blocks = extract_text_blocks(preprocessed, confidence_threshold=90)
        assert len(all_blocks) >= len(high_blocks)

    def test_empty_image_returns_empty(self) -> None:
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        preprocessed = preprocess(img)
        blocks = extract_text_blocks(preprocessed)
        assert blocks == []


class TestExtractFullText:
    def test_extracts_known_text(self) -> None:
        img = _make_text_image("Hello World", font_scale=1.5)
        preprocessed = preprocess(img)
        text = extract_full_text(preprocessed)
        # OCR should get at least some of the words
        assert "Hello" in text or "World" in text

    def test_sample_image_accuracy(self) -> None:
        """Verify OCR on the generated sample image extracts key phrases."""
        img = cv2.imread("assets/sample_text.png")
        assert img is not None, "assets/sample_text.png must exist"
        preprocessed = preprocess(img)
        text = extract_full_text(preprocessed)

        # These key phrases must appear in extracted text
        assert "Project Specifications" in text
        assert "2500" in text
        assert "March 2025" in text
        assert "Material Schedule" in text
        assert "Notes" in text

    def test_empty_image_returns_empty(self) -> None:
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        preprocessed = preprocess(img)
        text = extract_full_text(preprocessed)
        assert text == ""

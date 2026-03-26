"""Tests for phase2_ocr_pipeline.preprocess."""

import cv2
import numpy as np
import pytest

from phase2_ocr_pipeline.preprocess import (
    adaptive_threshold,
    denoise,
    deskew,
    detect_skew_angle,
    preprocess,
    to_grayscale,
)


def _make_white_image(width: int = 400, height: int = 300) -> np.ndarray:
    return np.ones((height, width, 3), dtype=np.uint8) * 255


class TestToGrayscale:
    def test_converts_bgr(self) -> None:
        img = _make_white_image()
        gray = to_grayscale(img)
        assert len(gray.shape) == 2
        assert gray.shape == (300, 400)

    def test_passthrough_if_already_gray(self) -> None:
        gray = np.ones((300, 400), dtype=np.uint8) * 128
        result = to_grayscale(gray)
        assert result is gray


class TestDenoise:
    def test_output_same_shape(self) -> None:
        gray = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
        result = denoise(gray)
        assert result.shape == gray.shape

    def test_reduces_noise(self) -> None:
        clean = np.ones((100, 100), dtype=np.uint8) * 200
        noisy = clean.copy()
        rng = np.random.default_rng(42)
        noisy = np.clip(noisy.astype(np.int16) + rng.integers(-50, 50, noisy.shape), 0, 255).astype(np.uint8)
        denoised = denoise(noisy)
        # Denoised should be closer to clean than noisy was
        noise_before = np.mean(np.abs(noisy.astype(float) - clean.astype(float)))
        noise_after = np.mean(np.abs(denoised.astype(float) - clean.astype(float)))
        assert noise_after < noise_before


class TestAdaptiveThreshold:
    def test_output_is_binary(self) -> None:
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = adaptive_threshold(gray)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_output_same_dimensions(self) -> None:
        gray = np.ones((200, 300), dtype=np.uint8) * 128
        result = adaptive_threshold(gray)
        assert result.shape == (200, 300)


class TestDeskew:
    def test_no_rotation_for_straight_text(self) -> None:
        img = np.ones((200, 400), dtype=np.uint8) * 255
        cv2.putText(img, "Straight text", (50, 100),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        angle = detect_skew_angle(img)
        assert abs(angle) < 5.0

    def test_deskew_preserves_dimensions(self) -> None:
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        result = deskew(img, angle=0.0)
        assert result.shape == img.shape

    def test_deskew_with_explicit_angle(self) -> None:
        img = np.ones((200, 400), dtype=np.uint8) * 255
        cv2.putText(img, "Text", (50, 100),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        result = deskew(img, angle=5.0)
        assert result.shape == img.shape


class TestPreprocess:
    def test_full_pipeline_returns_binary(self) -> None:
        img = _make_white_image()
        cv2.putText(img, "Hello World", (50, 150),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        result = preprocess(img)
        assert len(result.shape) == 2
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_full_pipeline_preserves_dimensions(self) -> None:
        img = _make_white_image(640, 480)
        result = preprocess(img)
        assert result.shape == (480, 640)

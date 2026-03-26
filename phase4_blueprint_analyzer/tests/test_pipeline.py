"""Tests for phase4_blueprint_analyzer.pipeline and layers."""

import cv2
import numpy as np
import pytest

from phase4_blueprint_analyzer import shape_layer, text_layer
from phase4_blueprint_analyzer.pipeline import _run_stage, analyze_image


def _make_test_image() -> np.ndarray:
    """Create a simple image with a rectangle and text."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 0), 2)
    cv2.putText(img, "Test Room", (70, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


class TestRunStage:
    def test_successful_stage(self) -> None:
        result, elapsed, error = _run_stage("test", lambda: [1, 2, 3])
        assert result == [1, 2, 3]
        assert elapsed >= 0
        assert error is None

    def test_failing_stage(self) -> None:
        def bad_fn() -> None:
            raise ValueError("boom")

        result, elapsed, error = _run_stage("test", bad_fn)
        assert result is None
        assert elapsed >= 0
        assert error is not None
        assert "ValueError" in error
        assert "boom" in error


class TestShapeLayer:
    def test_returns_list(self) -> None:
        img = _make_test_image()
        result = shape_layer.run(img)
        assert isinstance(result, list)

    def test_shape_dict_fields(self) -> None:
        img = _make_test_image()
        result = shape_layer.run(img)
        if result:
            shape = result[0]
            assert "shape_type" in shape
            assert "bbox" in shape
            assert "center" in shape


class TestTextLayer:
    def test_returns_dict_with_fields(self) -> None:
        img = _make_test_image()
        result = text_layer.run(img)
        assert isinstance(result, dict)
        assert "full_text" in result
        assert "text_blocks" in result
        assert "text_regions" in result
        assert "tables" in result

    def test_extracts_some_text(self) -> None:
        img = _make_test_image()
        result = text_layer.run(img)
        # Should find at least some text from "Test Room"
        assert result["text_blocks"] > 0 or len(result["full_text"]) > 0


class TestAnalyzeImage:
    def test_returns_all_sections(self) -> None:
        img = _make_test_image()
        result = analyze_image(img, page_num=1, yolo_weights="nonexistent.pt")
        assert result["page"] == 1
        assert "shapes" in result
        assert "text" in result
        assert "symbols" in result
        assert "timing" in result
        assert "errors" in result or isinstance(result.get("errors"), dict)

    def test_graceful_failure_on_missing_weights(self) -> None:
        """Symbol layer should not crash the pipeline if weights are missing."""
        img = _make_test_image()
        result = analyze_image(img, page_num=1, yolo_weights="nonexistent.pt")
        # Pipeline should complete — symbols empty but no crash
        assert result["page"] == 1
        assert isinstance(result["shapes"], list)
        assert isinstance(result["symbols"], list)

    def test_timing_present(self) -> None:
        img = _make_test_image()
        result = analyze_image(img, page_num=1, yolo_weights="nonexistent.pt")
        assert "shapes_sec" in result["timing"]
        assert "text_sec" in result["timing"]
        assert "symbols_sec" in result["timing"]

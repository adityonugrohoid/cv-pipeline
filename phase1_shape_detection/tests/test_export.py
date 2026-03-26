"""Tests for phase1_shape_detection.export."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from phase1_shape_detection.detector import Shape
from phase1_shape_detection.export import export_json, shapes_to_dict


def _make_shape(shape_type: str = "rectangle", **kwargs: object) -> Shape:
    defaults = {
        "contour": np.array([[[10, 10]], [[100, 10]], [[100, 80]], [[10, 80]]]),
        "bbox": (10, 10, 90, 70),
        "center": (55, 45),
        "properties": {"width": 90, "height": 70, "area": 6300.0, "aspect_ratio": 1.29},
    }
    defaults.update(kwargs)
    return Shape(shape_type=shape_type, **defaults)


class TestShapesToDict:
    def test_basic_structure(self) -> None:
        shapes = [_make_shape("rectangle"), _make_shape("circle")]
        data = shapes_to_dict(shapes)
        assert "shapes" in data
        assert "summary" in data
        assert data["summary"]["total"] == 2

    def test_shape_record_fields(self) -> None:
        shapes = [_make_shape("rectangle")]
        data = shapes_to_dict(shapes)
        record = data["shapes"][0]
        assert record["shape_type"] == "rectangle"
        assert record["center_x"] == 55
        assert record["center_y"] == 45
        assert record["bbox_x"] == 10
        assert record["bbox_y"] == 10
        assert record["width"] == 90
        assert record["height"] == 70
        assert record["confidence"] == 1.0

    def test_summary_counts_by_type(self) -> None:
        shapes = [
            _make_shape("rectangle"),
            _make_shape("circle"),
            _make_shape("rectangle"),
            _make_shape("triangle"),
        ]
        data = shapes_to_dict(shapes)
        assert data["summary"]["total"] == 4
        assert data["summary"]["by_type"]["rectangle"] == 2
        assert data["summary"]["by_type"]["circle"] == 1
        assert data["summary"]["by_type"]["triangle"] == 1

    def test_empty_shapes(self) -> None:
        data = shapes_to_dict([])
        assert data["shapes"] == []
        assert data["summary"]["total"] == 0

    def test_properties_merged_into_record(self) -> None:
        shape = _make_shape("circle", properties={"radius": 50, "area": 7854.0, "circularity": 0.95})
        data = shapes_to_dict([shape])
        record = data["shapes"][0]
        assert record["radius"] == 50
        assert record["circularity"] == 0.95


class TestExportJson:
    def test_writes_valid_json(self) -> None:
        shapes = [_make_shape("rectangle")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.json"
            result = export_json(shapes, path)

            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == result
            assert loaded["summary"]["total"] == 1

    def test_creates_parent_dirs(self) -> None:
        shapes = [_make_shape("rectangle")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "out.json"
            export_json(shapes, path)
            assert path.exists()

    def test_json_is_serializable(self) -> None:
        """Ensure no numpy types leak into JSON output."""
        shapes = [_make_shape("rectangle")]
        data = shapes_to_dict(shapes)
        # json.dumps will raise TypeError if non-serializable types are present
        json.dumps(data)

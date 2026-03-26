"""Tests for phase4_blueprint_analyzer.report."""

import json
import tempfile
from pathlib import Path

import pytest

from phase4_blueprint_analyzer.report import export_report, generate_report


def _make_page_result(page: int = 1) -> dict:
    return {
        "page": page,
        "shapes": [
            {"shape_type": "rectangle", "bbox": [0, 0, 100, 50], "center": [50, 25], "properties": {}},
            {"shape_type": "circle", "bbox": [200, 200, 40, 40], "center": [220, 220], "properties": {}},
        ],
        "text": {
            "full_text": "Hello World",
            "text_blocks": 2,
            "text_regions": [{"text": "Hello World", "bbox": [10, 10, 100, 20], "orientation": "horizontal"}],
            "tables": [],
        },
        "symbols": [
            {"class": "arrow", "confidence": 0.95, "bbox": [300, 300, 50, 50], "center": [325, 325]},
        ],
        "timing": {"shapes_sec": 0.1, "text_sec": 0.2, "symbols_sec": 0.3},
        "errors": {},
    }


class TestGenerateReport:
    def test_basic_structure(self) -> None:
        report = generate_report([_make_page_result()])
        assert "pages" in report
        assert "summary" in report
        assert report["summary"]["total_pages"] == 1

    def test_shape_summary(self) -> None:
        report = generate_report([_make_page_result()])
        shapes = report["summary"]["shapes"]
        assert shapes["total"] == 2
        assert shapes["by_type"]["rectangle"] == 1
        assert shapes["by_type"]["circle"] == 1

    def test_text_summary(self) -> None:
        report = generate_report([_make_page_result()])
        text = report["summary"]["text"]
        assert text["total_blocks"] == 2
        assert text["total_tables"] == 0

    def test_symbol_summary(self) -> None:
        report = generate_report([_make_page_result()])
        symbols = report["summary"]["symbols"]
        assert symbols["total"] == 1
        assert symbols["by_class"]["arrow"] == 1

    def test_multi_page(self) -> None:
        pages = [_make_page_result(1), _make_page_result(2)]
        report = generate_report(pages)
        assert report["summary"]["total_pages"] == 2
        assert report["summary"]["shapes"]["total"] == 4
        assert report["summary"]["symbols"]["total"] == 2

    def test_empty_pages(self) -> None:
        report = generate_report([])
        assert report["summary"]["total_pages"] == 0
        assert report["summary"]["shapes"]["total"] == 0

    def test_errors_included(self) -> None:
        page = _make_page_result()
        page["errors"] = {"symbols": "WeightNotFound: missing file"}
        report = generate_report([page])
        assert "errors" in report
        assert "symbols" in report["errors"]


class TestExportReport:
    def test_writes_valid_json(self) -> None:
        report = generate_report([_make_page_result()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            export_report(report, path)
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["summary"]["total_pages"] == 1

    def test_creates_parent_dirs(self) -> None:
        report = generate_report([_make_page_result()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "report.json"
            export_report(report, path)
            assert path.exists()

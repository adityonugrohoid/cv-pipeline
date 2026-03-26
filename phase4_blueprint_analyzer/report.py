"""Aggregate pipeline results into a structured takeoff report."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_report(page_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-page results into a final report.

    Args:
        page_results: List of per-page dictionaries from pipeline.analyze_pdf.

    Returns:
        Complete report with page-level data and summary statistics.
    """
    total_shapes = 0
    shape_type_counts: Counter[str] = Counter()
    total_text_blocks = 0
    total_tables = 0
    total_symbols = 0
    symbol_class_counts: Counter[str] = Counter()
    all_errors: dict[str, list[str]] = {}

    for page in page_results:
        # Shapes
        shapes = page.get("shapes", [])
        total_shapes += len(shapes)
        for s in shapes:
            shape_type_counts[s["shape_type"]] += 1

        # Text
        text_data = page.get("text", {})
        total_text_blocks += text_data.get("text_blocks", 0)
        total_tables += len(text_data.get("tables", []))

        # Symbols
        symbols = page.get("symbols", [])
        total_symbols += len(symbols)
        for sym in symbols:
            symbol_class_counts[sym["class"]] += 1

        # Errors
        for stage, err in page.get("errors", {}).items():
            all_errors.setdefault(stage, []).append(f"page {page['page']}: {err}")

    report: dict[str, Any] = {
        "pages": page_results,
        "summary": {
            "total_pages": len(page_results),
            "shapes": {
                "total": total_shapes,
                "by_type": dict(shape_type_counts),
            },
            "text": {
                "total_blocks": total_text_blocks,
                "total_tables": total_tables,
            },
            "symbols": {
                "total": total_symbols,
                "by_class": dict(symbol_class_counts),
            },
        },
    }

    if all_errors:
        report["errors"] = all_errors

    return report


def export_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a report to a JSON file.

    Args:
        report: Report dictionary from generate_report.
        output_path: Path to write JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report exported → %s", output_path)

"""Convert shape detection results to structured JSON."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .detector import Shape


def shapes_to_dict(shapes: list[Shape]) -> dict[str, Any]:
    """Convert a list of Shape objects to a serializable dictionary.

    Returns:
        Dictionary with 'shapes' list and 'summary' stats.
    """
    shape_records = []
    for shape in shapes:
        record: dict[str, Any] = {
            "shape_type": shape.shape_type,
            "center_x": shape.center[0],
            "center_y": shape.center[1],
            "bbox_x": shape.bbox[0],
            "bbox_y": shape.bbox[1],
            "width": shape.bbox[2],
            "height": shape.bbox[3],
            "confidence": shape.confidence,
        }
        # Merge shape-specific properties
        for key, value in shape.properties.items():
            if key not in record:
                record[key] = value
        shape_records.append(record)

    type_counts = Counter(s.shape_type for s in shapes)

    return {
        "shapes": shape_records,
        "summary": {
            "total": len(shapes),
            "by_type": dict(type_counts),
        },
    }


def export_json(shapes: list[Shape], output_path: str | Path) -> dict[str, Any]:
    """Export detection results to a JSON file.

    Args:
        shapes: List of detected Shape objects.
        output_path: Path to write JSON file.

    Returns:
        The serialized dictionary (same as written to file).
    """
    data = shapes_to_dict(shapes)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return data

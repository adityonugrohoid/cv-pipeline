"""Group nearby text blocks into logical text regions."""

import logging
from dataclasses import dataclass

from .ocr_engine import TextBlock

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A group of nearby text blocks forming a logical region."""

    text: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    blocks: list[TextBlock]
    orientation: str  # "horizontal" or "vertical"


def _merge_bbox(blocks: list[TextBlock]) -> tuple[int, int, int, int]:
    """Compute the bounding box that encloses all blocks."""
    x_min = min(b.bbox[0] for b in blocks)
    y_min = min(b.bbox[1] for b in blocks)
    x_max = max(b.bbox[0] + b.bbox[2] for b in blocks)
    y_max = max(b.bbox[1] + b.bbox[3] for b in blocks)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def _detect_orientation(blocks: list[TextBlock]) -> str:
    """Detect whether a group of blocks runs horizontally or vertically.

    Compares the horizontal span to the vertical span of the group.
    """
    if len(blocks) <= 1:
        return "horizontal"

    bbox = _merge_bbox(blocks)
    _, _, w, h = bbox
    return "vertical" if h > 3 * w else "horizontal"


def group_into_regions(
    blocks: list[TextBlock],
    y_gap_threshold: float = 20.0,
) -> list[TextRegion]:
    """Group text blocks into regions by y-coordinate proximity.

    Blocks on roughly the same line (within y_gap_threshold pixels) are
    grouped together. A new region starts when the vertical gap between
    consecutive blocks exceeds the threshold.

    Args:
        blocks: List of TextBlock objects, typically from ocr_engine.
        y_gap_threshold: Maximum vertical pixel gap to consider blocks
            part of the same region.

    Returns:
        List of TextRegion objects.
    """
    if not blocks:
        return []

    # Sort by y then x so we process top-to-bottom, left-to-right
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    regions: list[TextRegion] = []
    current_group: list[TextBlock] = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        prev = current_group[-1]
        prev_bottom = prev.bbox[1] + prev.bbox[3]
        curr_top = block.bbox[1]

        if curr_top - prev_bottom > y_gap_threshold:
            # Gap too large — flush current group as a region
            _flush_region(current_group, regions)
            current_group = [block]
        else:
            current_group.append(block)

    # Flush last group
    _flush_region(current_group, regions)

    logger.info("Grouped %d blocks into %d regions", len(blocks), len(regions))
    return regions


def _flush_region(group: list[TextBlock], regions: list[TextRegion]) -> None:
    """Create a TextRegion from a group of blocks and append to regions."""
    # Sort by x within the line for correct reading order
    line_sorted = sorted(group, key=lambda b: b.bbox[0])
    text = " ".join(b.text for b in line_sorted)
    bbox = _merge_bbox(line_sorted)
    orientation = _detect_orientation(line_sorted)
    regions.append(TextRegion(
        text=text,
        bbox=bbox,
        blocks=line_sorted,
        orientation=orientation,
    ))

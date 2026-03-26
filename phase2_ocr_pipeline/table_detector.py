"""Detect tables from horizontal/vertical line intersections and extract cell contents."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)


@dataclass
class Table:
    """A detected table with its grid structure and cell contents."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    rows: int
    cols: int
    cells: list[list[str]]  # cells[row][col] = text content


def _detect_lines(
    binary: np.ndarray,
    horizontal_scale: int = 30,
    vertical_scale: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect horizontal and vertical lines using morphological operations.

    Args:
        binary: Binary (thresholded) image.
        horizontal_scale: Kernel width divisor for horizontal line detection.
            Larger = only detect longer lines.
        vertical_scale: Kernel height divisor for vertical line detection.

    Returns:
        Tuple of (horizontal_mask, vertical_mask).
    """
    # Invert so lines are white on black
    if np.mean(binary) > 127:
        inv = cv2.bitwise_not(binary)
    else:
        inv = binary.copy()

    h, w = inv.shape[:2]

    # Horizontal lines
    h_kernel_len = max(w // horizontal_scale, 1)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # Vertical lines
    v_kernel_len = max(h // vertical_scale, 1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=2)

    return h_mask, v_mask


def _find_intersections(
    h_mask: np.ndarray,
    v_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Find intersection points between horizontal and vertical lines.

    Returns:
        List of (x, y) intersection coordinates, sorted top-to-bottom then left-to-right.
    """
    combined = cv2.bitwise_and(h_mask, v_mask)

    # Dilate intersections so nearby points merge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        m = cv2.moments(cnt)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            points.append((cx, cy))

    # Sort by y then x
    points.sort(key=lambda p: (p[1], p[0]))
    return points


def _cluster_values(values: list[int], gap: int = 15) -> list[int]:
    """Cluster nearby coordinate values and return cluster centers.

    Groups values that are within `gap` pixels of each other.
    """
    if not values:
        return []

    sorted_vals = sorted(values)
    clusters: list[list[int]] = [[sorted_vals[0]]]

    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    return [int(np.mean(c)) for c in clusters]


def _build_grid(
    points: list[tuple[int, int]],
    gap: int = 15,
) -> tuple[list[int], list[int]]:
    """Build a row/column grid from intersection points.

    Returns:
        Tuple of (row_ys, col_xs) — sorted lists of y and x coordinates
        defining the grid lines.
    """
    if not points:
        return [], []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    col_xs = _cluster_values(xs, gap)
    row_ys = _cluster_values(ys, gap)

    return row_ys, col_xs


def _extract_cell_text(
    image: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    lang: str = "eng",
) -> str:
    """Extract text from a single cell region using Tesseract.

    Adds a small padding inward to avoid reading line borders.
    """
    pad = 4
    x1c = min(x1 + pad, x2)
    y1c = min(y1 + pad, y2)
    x2c = max(x2 - pad, x1c)
    y2c = max(y2 - pad, y1c)

    cell = image[y1c:y2c, x1c:x2c]
    if cell.size == 0:
        return ""

    text = pytesseract.image_to_string(cell, lang=lang, config="--psm 7")
    return text.strip()


def detect_tables(
    image: np.ndarray,
    binary: np.ndarray | None = None,
    lang: str = "eng",
) -> list[Table]:
    """Detect tables in an image and extract cell contents.

    Args:
        image: Original image (grayscale or BGR) for OCR within cells.
        binary: Pre-binarized version of the image. If None, Otsu is applied.
        lang: Tesseract language code.

    Returns:
        List of detected Table objects.
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if binary is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h_mask, v_mask = _detect_lines(binary)
    points = _find_intersections(h_mask, v_mask)

    if len(points) < 4:
        logger.info("Found %d intersections — not enough for a table", len(points))
        return []

    row_ys, col_xs = _build_grid(points)

    if len(row_ys) < 2 or len(col_xs) < 2:
        logger.info("Grid too small (%d rows, %d cols)", len(row_ys), len(col_xs))
        return []

    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1

    cells: list[list[str]] = []
    for r in range(n_rows):
        row: list[str] = []
        for c in range(n_cols):
            text = _extract_cell_text(
                gray,
                col_xs[c], row_ys[r],
                col_xs[c + 1], row_ys[r + 1],
                lang=lang,
            )
            row.append(text)
        cells.append(row)

    x = col_xs[0]
    y = row_ys[0]
    w = col_xs[-1] - col_xs[0]
    h = row_ys[-1] - row_ys[0]

    table = Table(bbox=(x, y, w, h), rows=n_rows, cols=n_cols, cells=cells)
    logger.info("Detected table: %d rows x %d cols at (%d, %d)", n_rows, n_cols, x, y)
    return [table]

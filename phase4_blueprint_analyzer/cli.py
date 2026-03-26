"""CLI entrypoint for Phase 4 blueprint analyzer.

Usage:
    python -m phase4_blueprint_analyzer.cli analyze --input blueprint.pdf --output report.json
    python -m phase4_blueprint_analyzer.cli generate  # create sample blueprint PDF
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_blueprint(output_path: str = "assets/sample_blueprint.pdf") -> None:
    """Generate a sample blueprint PDF with shapes, text, and symbols."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter  # 612 x 792 points

    # --- Title block ---
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, h - 50, "Floor Plan — Building A")

    c.setFont("Helvetica", 12)
    c.drawString(50, h - 75, "Project: CV Pipeline Demo")
    c.drawString(50, h - 92, "Date: March 2025")
    c.drawString(50, h - 109, "Scale: 1/4\" = 1'-0\"")

    # --- Room outlines (rectangles) ---
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(2)

    # Main room
    c.rect(50, 300, 250, 200)
    c.setFont("Helvetica", 10)
    c.drawString(130, 390, "Office 101")
    c.drawString(120, 370, "25'-0\" x 20'-0\"")

    # Second room
    c.rect(300, 300, 200, 200)
    c.drawString(360, 390, "Office 102")
    c.drawString(350, 370, "20'-0\" x 20'-0\"")

    # Hallway
    c.rect(50, 220, 450, 80)
    c.drawString(230, 255, "Corridor")

    # --- Dimension lines ---
    c.setLineWidth(1)
    c.setStrokeColorRGB(0.3, 0.3, 0.3)

    # Horizontal dimension for main room
    y_dim = 520
    c.line(50, y_dim, 300, y_dim)
    c.line(50, y_dim - 5, 50, y_dim + 5)
    c.line(300, y_dim - 5, 300, y_dim + 5)
    c.setFont("Helvetica", 9)
    c.drawString(155, y_dim + 5, "25'-0\"")

    # Horizontal dimension for second room
    c.line(300, y_dim, 500, y_dim)
    c.line(500, y_dim - 5, 500, y_dim + 5)
    c.drawString(380, y_dim + 5, "20'-0\"")

    # --- Door swings (quarter arcs) ---
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(1.5)

    # Door in main room (bottom-left of room)
    import math
    door_x, door_y = 170, 300
    door_r = 30
    c.arc(door_x - door_r, door_y - door_r, door_x + door_r, door_y + door_r, 0, 90)
    c.line(door_x, door_y, door_x + door_r, door_y)
    c.line(door_x, door_y, door_x, door_y + door_r)

    # Door in second room
    door_x2, door_y2 = 380, 300
    c.arc(door_x2 - door_r, door_y2 - door_r, door_x2 + door_r, door_y2 + door_r, 0, 90)
    c.line(door_x2, door_y2, door_x2 + door_r, door_y2)
    c.line(door_x2, door_y2, door_x2, door_y2 + door_r)

    # --- Electrical outlet symbols ---
    c.setLineWidth(1)
    for ox, oy in [(80, 420), (270, 420), (330, 420), (470, 420)]:
        r = 8
        c.circle(ox, oy, r)
        c.line(ox - 3, oy - 4, ox - 3, oy + 4)
        c.line(ox + 3, oy - 4, ox + 3, oy + 4)

    # --- Arrows ---
    c.setLineWidth(1.5)
    # North arrow
    c.drawString(520, h - 130, "N")
    c.line(530, h - 140, 530, h - 180)
    c.line(530, h - 140, 525, h - 150)
    c.line(530, h - 140, 535, h - 150)

    # --- Notes section ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 180, "Notes:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 162, "1. All dimensions are to face of stud.")
    c.drawString(50, 147, "2. Verify all dimensions on site.")
    c.drawString(50, 132, "3. Electrical outlets at 18\" AFF.")

    # --- Material table ---
    c.setFont("Helvetica-Bold", 10)
    c.drawString(320, 180, "Material Takeoff")

    table_x = 320
    table_y = 50
    col_w = [120, 60, 80]
    row_h = 20
    headers = ["Material", "Qty", "Unit"]
    rows = [
        ["Drywall", "120", "sheets"],
        ["Paint", "15", "gallons"],
        ["Carpet", "500", "sq ft"],
    ]

    c.setFont("Helvetica", 9)
    c.setLineWidth(1)
    for r_idx in range(len(rows) + 2):
        y = table_y + r_idx * row_h
        c.line(table_x, y, table_x + sum(col_w), y)
    x = table_x
    for cw in col_w:
        c.line(x, table_y, x, table_y + (len(rows) + 1) * row_h)
        x += cw
    c.line(x, table_y, x, table_y + (len(rows) + 1) * row_h)

    c.setFont("Helvetica-Bold", 9)
    x = table_x
    for i, hdr in enumerate(headers):
        c.drawString(x + 5, table_y + len(rows) * row_h + 5, hdr)
        x += col_w[i]

    c.setFont("Helvetica", 9)
    for r_idx, row in enumerate(rows):
        x = table_x
        y = table_y + (len(rows) - 1 - r_idx) * row_h + 5
        for c_idx, cell in enumerate(row):
            c.drawString(x + 5, y, cell)
            x += col_w[c_idx]

    c.save()
    logger.info("Generated sample blueprint → %s", path)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the full blueprint analysis pipeline."""
    from .pipeline import analyze_pdf
    from .report import export_report, generate_report

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    page_results = analyze_pdf(
        input_path,
        yolo_weights=args.weights,
    )

    report = generate_report(page_results)
    export_report(report, args.output)

    summary = report["summary"]
    logger.info("=== Analysis Summary ===")
    logger.info("Pages: %d", summary["total_pages"])
    logger.info("Shapes: %d (%s)", summary["shapes"]["total"],
                ", ".join(f"{k}={v}" for k, v in summary["shapes"]["by_type"].items()))
    logger.info("Text blocks: %d, Tables: %d",
                summary["text"]["total_blocks"], summary["text"]["total_tables"])
    logger.info("Symbols: %d (%s)", summary["symbols"]["total"],
                ", ".join(f"{k}={v}" for k, v in summary["symbols"]["by_class"].items()))

    if "errors" in report:
        logger.warning("Errors: %s", report["errors"])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phase4_blueprint_analyzer",
        description="Multi-stage blueprint analysis pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a blueprint PDF")
    p_analyze.add_argument("--input", required=True, help="Input PDF path")
    p_analyze.add_argument("--output", default="outputs/report.json", help="Output JSON report path")
    p_analyze.add_argument("--weights", default="models/best.pt", help="YOLO weights path")

    # generate
    p_gen = sub.add_parser("generate", help="Generate sample blueprint PDF")
    p_gen.add_argument("--output", default="assets/sample_blueprint.pdf",
                       help="Output path for generated PDF")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "generate":
        generate_sample_blueprint(args.output)


if __name__ == "__main__":
    main()

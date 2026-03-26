"""FastAPI server for blueprint analysis.

POST /analyze — upload a PDF and receive a JSON analysis report.
GET  /health  — health check.
Swagger UI at /docs.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .pipeline import analyze_pdf
from .report import generate_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Blueprint Analyzer",
    description="Multi-stage construction blueprint analysis: shapes, text/OCR, symbol detection.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    yolo_weights: str = "models/best.pt",
) -> JSONResponse:
    """Upload a PDF and receive a full analysis report.

    The pipeline runs three detection stages per page:
    - Shape detection (Phase 1)
    - OCR text extraction (Phase 2)
    - YOLO symbol detection (Phase 3)
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are accepted."},
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        page_results = analyze_pdf(tmp_path, yolo_weights=yolo_weights)
        report = generate_report(page_results)
        return JSONResponse(content=report)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

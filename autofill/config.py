
"""Central configuration. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DB_PATH = Path(
    os.environ.get(
        "RESUME_DB_PATH",
        str(PROJECT_DIR / "resume_history.db"),
    )
)

# Maximum screenshots per candidate run.
MAX_UPLOADS = int(os.environ.get("RESUME_MAX_UPLOADS", "30"))

# Image validation limits.
MAX_IMAGE_BYTES = int(os.environ.get("RESUME_MAX_IMAGE_MB", "20")) * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "pdf")

# PDF processing settings.
MAX_PDF_PAGES = int(os.environ.get("RESUME_MAX_PDF_PAGES", "10"))
PDF_RENDER_SCALE = float(os.environ.get("RESUME_PDF_SCALE", "2.0"))

# OCR image resizing settings.
OCR_MAX_EDGE = int(os.environ.get("RESUME_OCR_MAX_EDGE", "1280"))
OCR_MIN_WIDTH = 800

# OCR confidence settings.
OCR_MIN_SCORE = float(os.environ.get("RESUME_OCR_MIN_SCORE", "0.30"))
OCR_REVIEW_SCORE = 0.80
OCR_CACHE_SIZE = 64

# Database retention: 0 = keep only records processed today.
RETENTION_DAYS = int(os.environ.get("RESUME_RETENTION_DAYS", "0"))

# OCR speed/accuracy profile: "fast" or "accurate".
OCR_PROFILE = os.environ.get(
    "RESUME_OCR_PROFILE", "fast"
).strip().lower()

# Enable only when using the compatible PaddlePaddle version.
OCR_USE_MKLDNN = os.environ.get("RESUME_OCR_MKLDNN", "0") == "1"

# Maximum CPU threads used by OCR.
OCR_CPU_THREADS = int(
    os.environ.get(
        "RESUME_OCR_THREADS",
        str(os.cpu_count() or 4),
    )
)

# OCR backend:
# "rapidocr" uses ONNX Runtime and is already listed in requirements.txt.
# "paddle" requires paddleocr and paddlepaddle to be installed.
# "auto" tries RapidOCR first, then PaddleOCR.
#
# RapidOCR is the default to avoid importing the missing PaddleOCR package
# in the Streamlit Cloud environment.
OCR_BACKEND = os.environ.get(
    "RESUME_OCR_BACKEND", "rapidocr"
).strip().lower()

# Number of concurrent OCR workers (RapidOCR only).
# Default: half the CPU cores, limited to 1–4 workers.
OCR_WORKERS = int(
    os.environ.get(
        "RESUME_OCR_WORKERS",
        str(max(1, min(4, (os.cpu_count() or 2) // 2))),
    )
)

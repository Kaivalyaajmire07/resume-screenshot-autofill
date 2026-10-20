"""Central configuration. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.environ.get("RESUME_DB_PATH", str(PROJECT_DIR / "resume_history.db")))

# Maximum screenshots per candidate run (unchanged from the original app).
MAX_UPLOADS = int(os.environ.get("RESUME_MAX_UPLOADS", "30"))

# Image validation limits.
MAX_IMAGE_BYTES = int(os.environ.get("RESUME_MAX_IMAGE_MB", "20")) * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "pdf")

# PDFs: pages read per file, and render scale for scanned pages (2.0 = 144 dpi).
MAX_PDF_PAGES = int(os.environ.get("RESUME_MAX_PDF_PAGES", "10"))
PDF_RENDER_SCALE = float(os.environ.get("RESUME_PDF_SCALE", "2.0"))

# Longest edge sent to OCR. Larger screenshots are downscaled for CPU speed.
OCR_MAX_EDGE = int(os.environ.get("RESUME_OCR_MAX_EDGE", "1280"))
# Narrow screenshots are upscaled to at least this width (tiny text OCRs badly).
OCR_MIN_WIDTH = 800
# OCR lines below this score are dropped (and counted in a warning).
OCR_MIN_SCORE = float(os.environ.get("RESUME_OCR_MIN_SCORE", "0.30"))
# OCR lines below this score make the extracted field "needs review".
OCR_REVIEW_SCORE = 0.80
OCR_CACHE_SIZE = 64

# Retention: 0 = keep only records processed today (original behaviour).
# N > 0 keeps today plus the previous N days.
RETENTION_DAYS = int(os.environ.get("RESUME_RETENTION_DAYS", "0"))

# OCR speed/accuracy profile: "fast" (mobile models, default) or "accurate" (server models + line orientation).
OCR_PROFILE = os.environ.get("RESUME_OCR_PROFILE", "fast").strip().lower()
# Set RESUME_OCR_MKLDNN=1 only if you installed paddlepaddle==3.0.0 (oneDNN is faster but crashes on 3.1+).
OCR_USE_MKLDNN = os.environ.get("RESUME_OCR_MKLDNN", "0") == "1"
OCR_CPU_THREADS = int(os.environ.get("RESUME_OCR_THREADS", str(os.cpu_count() or 4)))

# OCR backend: "auto" (RapidOCR if installed, else PaddleOCR), "rapidocr" or "paddle".
# RapidOCR runs the same PP-OCR models through ONNX Runtime: lighter, and safe to run in parallel.
OCR_BACKEND = os.environ.get("RESUME_OCR_BACKEND", "auto").strip().lower()
# Screenshots OCR'd at the same time (RapidOCR only). Default: half the cores, 1-4.
OCR_WORKERS = int(os.environ.get("RESUME_OCR_WORKERS", str(max(1, min(4, (os.cpu_count() or 2) // 2)))))

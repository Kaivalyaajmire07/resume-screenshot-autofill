"""Image validation and preprocessing (Pillow, with OpenCV used only if installed)."""
from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from . import config

try:  # OpenCV is optional: PaddleOCR normally installs it, but we never depend on it.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    cv2 = None


class ImageError(ValueError):
    """Raised for unreadable, oversized or unsupported images (message is user-facing)."""


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_image(data: bytes) -> Image.Image:
    """Validate and decode image bytes into an upright RGB image."""
    if not data:
        raise ImageError("empty file")
    if len(data) > config.MAX_IMAGE_BYTES:
        raise ImageError(f"file is larger than {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        probe = Image.open(BytesIO(data))
        probe.verify()  # cheap integrity check; invalidates the handle, so reopen
        img = Image.open(BytesIO(data))
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageError("not a readable image") from exc
    if img.width * img.height > config.MAX_IMAGE_PIXELS:
        raise ImageError("image dimensions are too large")
    if img.width < 20 or img.height < 20:
        raise ImageError("image is too small to contain text")
    try:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            img = img.convert("RGB")
    except OSError as exc:  # truncated files surface here
        raise ImageError("image data is corrupted") from exc
    return img


def _is_dark(img: Image.Image) -> bool:
    small = img.convert("L").resize((64, 64))
    return float(np.asarray(small).mean()) < 90.0


def preprocess(img: Image.Image) -> Image.Image:
    """Scale, invert dark-mode screenshots and gently boost contrast for OCR."""
    longest = max(img.size)
    if longest > config.OCR_MAX_EDGE:
        scale = config.OCR_MAX_EDGE / longest
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.Resampling.LANCZOS)
    elif img.width < config.OCR_MIN_WIDTH:
        scale = min(2.0, config.OCR_MIN_WIDTH / img.width)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    if _is_dark(img):
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img, cutoff=1)
    return ImageEnhance.Contrast(img).enhance(1.08)


def to_bgr_array(img: Image.Image) -> np.ndarray:
    """PaddleOCR expects a BGR uint8 array (OpenCV convention)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(arr[:, :, ::-1])


def thumbnail_png(data: bytes, max_size: int = 220) -> bytes:
    img = load_image(data)
    img.thumbnail((max_size, max_size))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def crop_percent(data: bytes, left: int, top: int, width: int, height: int) -> bytes:
    """Crop using percentages of the image size; returns PNG bytes."""
    img = load_image(data)
    w, h = img.size
    x1, y1 = int(w * left / 100), int(h * top / 100)
    x2 = min(w, int(w * (left + width) / 100))
    y2 = min(h, int(h * (top + height) / 100))
    cropped = img.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
    out = BytesIO()
    cropped.save(out, format="PNG")
    return out.getvalue()

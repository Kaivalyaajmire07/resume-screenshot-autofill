"""Run on the target machine to verify local OCR works:  python -m autofill.selfcheck

Renders a small test image, runs the real PaddleOCR engine on it and prints what it read.
The first run downloads the OCR models (needs internet once); later runs are offline.
"""
import sys
import time

from PIL import Image, ImageDraw, ImageFont

from .imaging import preprocess
from .ocr_engine import OcrEngine, OcrError

TEXT = ["Anita Rao", "Mobile: 9876543210", "Email: anita.rao@example.com"]


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    img = Image.new("RGB", (1100, 330), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(TEXT):
        draw.text((40, 30 + i * 90), line, fill="black", font=_font(40))
    started = time.perf_counter()
    try:
        engine = OcrEngine()
        lines = engine.recognize(preprocess(img))
        first = time.perf_counter() - started
        t2 = time.perf_counter()
        engine.recognize(preprocess(img))  # same image, no cache key -> real second inference
        steady = time.perf_counter() - t2
    except OcrError as exc:
        print("FAIL:", exc)
        return 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(f"First run (includes model load): {first:.1f}s | steady-state per screenshot: {steady:.2f}s")
    for ln in lines:
        print(f"  {ln.score:.2f}  {ln.text}")
    found = " ".join(l.text for l in lines).replace(" ", "").lower()
    ok = all(t.replace(" ", "").lower() in found for t in ("anita", "9876543210", "anita.rao@example.com"))
    print("PASS" if ok else "WARNING: OCR ran but the text did not match — check fonts/models")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

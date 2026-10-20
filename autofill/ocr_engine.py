"""Local PaddleOCR wrapper.

Fixes the Windows/CPU failure

    NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
    [pir::ArrayAttribute<pir::DoubleAttribute>] (onednn_instruction.cc:118)

which is raised by PaddlePaddle 3.x when the oneDNN (MKLDNN) CPU path runs a PIR program.
oneDNN is therefore disabled before and during engine construction. If a runtime error that
still mentions oneDNN occurs, the engine is rebuilt once with oneDNN forced off.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from PIL import Image

from . import config
from .imaging import to_bgr_array

# Must be set before paddle is imported anywhere in the process.
if not config.OCR_USE_MKLDNN:
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

log = logging.getLogger(__name__)

ONEDNN_HINT = (
    "PaddlePaddle's oneDNN CPU path is failing on this machine. Disable it (this app does) or "
    "install a known-good build: pip install paddlepaddle==3.0.0"
)


class OcrError(RuntimeError):
    """OCR could not run (message is user-facing)."""


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float = 1.0
    bbox: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 in preprocessed pixels
    source_index: int = 0  # 1-based screenshot number
    source_name: str = ""


def _poly_to_bbox(poly: Any) -> tuple[int, int, int, int] | None:
    try:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    except Exception:
        return None


def parse_paddle_result(result: Any) -> list[OcrLine]:
    """Convert PaddleOCR 3.x (and legacy 2.x) predict/ocr output into OcrLine objects."""
    lines: list[OcrLine] = []
    for page in result or []:
        data = getattr(page, "json", page)
        if callable(data):
            data = data()
        if isinstance(data, dict):
            res = data.get("res", data)
            if not isinstance(res, dict):
                continue
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            polys = res.get("rec_polys")
            if polys is None or len(polys) == 0:
                polys = res.get("dt_polys") or []
            for i, text in enumerate(texts):
                text = str(text).strip()
                if not text:
                    continue
                score = float(scores[i]) if i < len(scores) else 1.0
                bbox = _poly_to_bbox(polys[i]) if i < len(polys) else None
                lines.append(OcrLine(text=text, score=score, bbox=bbox))
        elif isinstance(page, (list, tuple)):  # legacy: [[poly, (text, score)], ...]
            for item in page:
                try:
                    text = str(item[1][0]).strip()
                    if text:
                        lines.append(OcrLine(text=text, score=float(item[1][1]), bbox=_poly_to_bbox(item[0])))
                except Exception:
                    continue
    return lines


def _is_onednn_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(t in msg for t in ("onednn", "mkldnn", "convertpirattribute2runtimeattribute"))


def _build_paddle_ocr():
    from paddleocr import PaddleOCR  # imported lazily: heavy

    try:  # belt and braces: also switch the flag at runtime
        import paddle

        paddle.set_flags({"FLAGS_use_mkldnn": bool(config.OCR_USE_MKLDNN)})
    except Exception:
        pass

    kwargs: dict[str, Any] = dict(
        lang="en",
        use_doc_orientation_classify=False,  # screenshots are upright; saves a model pass
        use_doc_unwarping=False,  # screenshots are flat; saves a model pass
        enable_mkldnn=bool(config.OCR_USE_MKLDNN),  # False avoids the PaddlePaddle 3.1+ oneDNN crash
        cpu_threads=config.OCR_CPU_THREADS,
        text_det_limit_side_len=config.OCR_MAX_EDGE,
        text_det_limit_type="max",
    )
    if config.OCR_PROFILE == "accurate":
        kwargs["use_textline_orientation"] = True
    else:  # fast: small mobile detector/recognizer, no per-line orientation model
        kwargs.update(use_textline_orientation=False,
                      text_detection_model_name="PP-OCRv5_mobile_det",
                      text_recognition_model_name="PP-OCRv5_mobile_rec")
    # If this PaddleOCR version rejects an option, drop it (least important first) and retry.
    droppable = ["cpu_threads", "text_det_limit_type", "text_det_limit_side_len", "text_detection_model_name",
                 "text_recognition_model_name", "enable_mkldnn", "use_doc_unwarping",
                 "use_doc_orientation_classify"]
    while True:
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as exc:
            dropped = next((k for k in droppable if k in kwargs), None)
            if dropped is None:
                raise OcrError(f"Could not start PaddleOCR: {exc}") from exc
            log.warning("PaddleOCR rejected %s (%s); retrying without it", dropped, exc)
            kwargs.pop(dropped)


class RapidOcrBackend:
    """RapidOCR (ONNX Runtime) with a PaddleOCR-shaped ``predict`` so the rest of the app is unchanged."""

    parallel_safe = True

    def __init__(self, threads_per_job: int):
        from rapidocr_onnxruntime import RapidOCR  # imported lazily

        try:
            self._ocr = RapidOCR(intra_op_num_threads=threads_per_job, inter_op_num_threads=1)
        except Exception:
            self._ocr = RapidOCR()

    def predict(self, array: Any) -> list[dict]:
        result, _ = self._ocr(array, use_cls=False)  # screenshots are upright: skip the angle model
        rows = result or []
        return [{"res": {"rec_texts": [r[1] for r in rows], "rec_scores": [float(r[2]) for r in rows],
                         "rec_polys": [r[0] for r in rows]}}]


def _rapidocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _default_factory():
    if config.OCR_BACKEND == "rapidocr" or (config.OCR_BACKEND == "auto" and _rapidocr_available()):
        threads = max(1, (os.cpu_count() or 2) // max(1, config.OCR_WORKERS))
        return RapidOcrBackend(threads)
    return _build_paddle_ocr()


class OcrCache:
    """Tiny thread-safe LRU so re-clicking Extract on the same screenshots is instant."""

    def __init__(self, size: int = config.OCR_CACHE_SIZE):
        self._size = size
        self._data: OrderedDict[str, list[OcrLine]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> list[OcrLine] | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return list(self._data[key])
        return None

    def put(self, key: str, lines: list[OcrLine]) -> None:
        with self._lock:
            self._data[key] = list(lines)
            self._data.move_to_end(key)
            while len(self._data) > self._size:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class OcrEngine:
    """Lazy, thread-safe PaddleOCR singleton. Pass ``factory`` to inject a fake in tests."""

    def __init__(self, factory: Callable[[], Any] | None = None):
        self._factory = factory or _default_factory
        self._ocr: Any = None
        self._lock = threading.Lock()
        self._init_lock = threading.RLock()
        self.cache = OcrCache()

    def _get(self):
        if self._ocr is None:
            with self._init_lock:
                if self._ocr is None:
                    self._ocr = self._factory()
        return self._ocr

    @property
    def parallel_safe(self) -> bool:
        return bool(getattr(self._get(), "parallel_safe", False))

    def warm_up(self) -> None:
        """Load models and run one tiny inference so the first real click is fast."""
        try:
            with self._lock:
                self._get().predict(to_bgr_array(Image.new("RGB", (320, 96), "white")))
        except Exception as exc:  # warm-up is best effort; real errors surface on use
            log.warning("OCR warm-up failed: %s", exc)

    def recognize(self, image: Image.Image, cache_key: str | None = None) -> list[OcrLine]:
        if cache_key:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit
        array = to_bgr_array(image)
        backend = self._get()
        if getattr(backend, "parallel_safe", False):  # ONNX sessions can be shared across threads
            lines = parse_paddle_result(backend.predict(array))
            if cache_key:
                self.cache.put(cache_key, lines)
            return lines
        with self._lock:  # Paddle inference objects are not safe for concurrent use
            try:
                result = self._get().predict(array)
            except Exception as exc:
                if not _is_onednn_error(exc):
                    raise
                log.warning("oneDNN failure (%s); rebuilding engine with oneDNN disabled", exc)
                os.environ["FLAGS_use_mkldnn"] = "0"
                self._ocr = None
                try:
                    result = self._get().predict(array)
                except Exception as exc2:
                    if _is_onednn_error(exc2):
                        raise OcrError(ONEDNN_HINT) from exc2
                    raise
        lines = parse_paddle_result(result)
        if cache_key:
            self.cache.put(cache_key, lines)
        return lines


_ENGINE: OcrEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> OcrEngine:
    """Process-wide engine (survives Streamlit reruns, loads models once)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = OcrEngine()
        return _ENGINE


def filter_lines(lines: Iterable[OcrLine], min_score: float = config.OCR_MIN_SCORE) -> tuple[list[OcrLine], int]:
    kept, dropped = [], 0
    for line in lines:
        if line.score < min_score:
            dropped += 1
        else:
            kept.append(line)
    return kept, dropped

import pytest
from PIL import Image

from autofill import ocr_engine
from autofill.ocr_engine import OcrEngine, OcrError, filter_lines, parse_paddle_result
from conftest import FakeOcr

IMG = Image.new("RGB", (200, 100), "white")
ONEDNN = "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]"


def test_parse_v3_dict_and_json_attribute():
    raw = {"res": {"rec_texts": ["Hello", " ", "World"], "rec_scores": [0.9, 0.9, 0.5],
                   "rec_polys": [[[1, 2], [5, 2], [5, 9], [1, 9]]] * 3}}

    class Page:  # PaddleOCR 3.x result objects expose .json
        json = raw
    for source in ([raw], [Page()]):
        lines = parse_paddle_result(source)
        assert [l.text for l in lines] == ["Hello", "World"]
        assert lines[0].bbox == (1, 2, 5, 9) and lines[1].score == 0.5


def test_parse_legacy_structure():
    legacy = [[[[[0, 0], [4, 0], [4, 4], [0, 4]], ("Old API", 0.8)]]]
    assert [l.text for l in parse_paddle_result(legacy)] == ["Old API"]


def test_filter_low_confidence():
    lines = parse_paddle_result([{"res": {"rec_texts": ["a", "b"], "rec_scores": [0.9, 0.1]}}])
    kept, dropped = filter_lines(lines)
    assert [l.text for l in kept] == ["a"] and dropped == 1


def test_onednn_failure_triggers_rebuild_and_recovers():
    built = []
    def factory():
        built.append(1)
        return FakeOcr(fail_times=1 if len(built) == 1 else 0, fail_msg=ONEDNN)
    eng = OcrEngine(factory=factory)
    assert len(eng.recognize(IMG)) > 0 and len(built) == 2


def test_persistent_onednn_failure_gives_actionable_error():
    eng = OcrEngine(factory=lambda: FakeOcr(fail_times=99, fail_msg=ONEDNN))
    with pytest.raises(OcrError, match="paddlepaddle==3.0.0"):
        eng.recognize(IMG)


def test_other_errors_are_not_swallowed():
    eng = OcrEngine(factory=lambda: FakeOcr(fail_times=1, fail_msg="something else"))
    with pytest.raises(NotImplementedError):
        eng.recognize(IMG)


def test_cache_avoids_second_inference():
    fake = FakeOcr(); eng = OcrEngine(factory=lambda: fake)
    eng.recognize(IMG, cache_key="k"); eng.recognize(IMG, cache_key="k")
    assert fake.calls == 1


def test_mkldnn_is_disabled_before_paddle_import():
    import os
    assert os.environ["FLAGS_use_mkldnn"] == "0"


def test_builder_passes_enable_mkldnn_false(monkeypatch):
    captured = {}
    class Fake:
        def __init__(self, **kw): captured.update(kw)
    import sys, types
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=Fake))
    ocr_engine._build_paddle_ocr()
    assert captured["enable_mkldnn"] is False and captured["lang"] == "en"


def test_builder_drops_unknown_kwargs(monkeypatch):
    calls = []
    class Strict:
        def __init__(self, **kw):
            calls.append(set(kw))
            if "enable_mkldnn" in kw or "use_doc_unwarping" in kw:
                raise TypeError("unexpected keyword")
    import sys, types
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=Strict))
    ocr_engine._build_paddle_ocr()
    assert "enable_mkldnn" not in calls[-1]

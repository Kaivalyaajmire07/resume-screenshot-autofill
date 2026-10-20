from datetime import date

from autofill import config, nlq
from autofill.attachments import AttachmentQueue
from autofill.history import save_candidate
from autofill.imaging import ImageError, load_image, preprocess, to_bgr_array
import pytest


def test_queue_add_dedupe_remove_move(png_bytes):
    q = AttachmentQueue()
    assert q.add("a.png", png_bytes("white"))[0] and q.add("b.png", png_bytes("black"))[0]
    ok, msg = q.add("dup.png", png_bytes("white"))
    assert not ok and "identical" in msg and len(q) == 2
    a, b = q.items
    q.move(b.id, -1)
    assert [i.name for i in q.items] == ["b.png", "a.png"]
    q.remove(a.id)
    assert [i.name for i in q.items] == ["b.png"]


def test_queue_rejects_invalid_and_over_limit(png_bytes, monkeypatch):
    q = AttachmentQueue()
    ok, msg = q.add("x.png", b"garbage")
    assert not ok and "not a readable image" in msg
    monkeypatch.setattr(config, "MAX_UPLOADS", 1)
    q.add("a.png", png_bytes("white"))
    ok, msg = q.add("b.png", png_bytes("black"))
    assert not ok and "limit" in msg


def test_crop_and_reset(png_bytes):
    q = AttachmentQueue(); q.add("a.png", png_bytes(size=(400, 200)))
    item = q.items[0]
    q.apply_crop(item.id, 0, 0, 50, 50)
    assert load_image(item.data).size == (200, 100) and item.cropped
    q.reset_crop(item.id)
    assert load_image(item.data).size == (400, 200) and not item.cropped


def test_preprocess_inverts_dark_and_returns_bgr(png_bytes):
    dark = preprocess(load_image(png_bytes("black", (1200, 600))))
    assert dark.getpixel((5, 5))[0] > 200            # dark-mode screenshot became dark-on-light
    assert to_bgr_array(dark).shape == (600, 1200, 3)


def test_huge_and_tiny_images_rejected():
    from io import BytesIO
    from PIL import Image
    buf = BytesIO(); Image.new("RGB", (5, 5)).save(buf, format="PNG")
    with pytest.raises(ImageError):
        load_image(buf.getvalue())


def test_large_screenshot_is_downscaled(png_bytes):
    out = preprocess(load_image(png_bytes(size=(4000, 3000))))
    assert max(out.size) == config.OCR_MAX_EDGE


def _seed(repo):
    save_candidate(repo, {"first_name": "Anita", "last_name": "Rao", "email": "a@x.com", "mobile": "9876543210",
                          "key_skills": "• Python\n• SQL", "city": "Pune"}, "r", 1)
    save_candidate(repo, {"first_name": "Ravi", "last_name": "Kumar", "key_skills": "• Java", "city": "Delhi"}, "r", 1)


def test_nlq_is_read_only_and_answers(repo):
    _seed(repo)
    assert "2 candidate" in nlq.answer(repo, "how many candidates today?").message
    ans = nlq.answer(repo, "email of anita")
    assert ans.rows == [{"Candidate": "Anita Rao", "Email": "a@x.com"}]
    assert [r["Candidate"] for r in nlq.answer(repo, "who knows python").rows] == ["Anita Rao"]
    assert [r["Candidate"] for r in nlq.answer(repo, "candidates from delhi").rows] == ["Ravi Kumar"]
    assert nlq.answer(repo, "kumar").rows[0]["Candidate"] == "Ravi Kumar"
    nlq.answer(repo, "'; DROP TABLE history; --")          # never interpreted as SQL
    assert repo.count(date.today()) == 2
    assert "couldn't" in nlq.answer(repo, "qwerty zzz").message

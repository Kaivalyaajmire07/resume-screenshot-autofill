"""Attachment queue for the screenshot composer (pure Python, no Streamlit)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from . import config
from . import pdfs
from .imaging import ImageError, crop_percent, image_hash, load_image, thumbnail_png

_ids = itertools.count(1)


@dataclass
class QueuedImage:
    id: int
    name: str
    data: bytes  # current bytes (cropped, if a crop was applied)
    original: bytes
    hash: str  # hash of the original bytes (dedupe key)
    thumb: bytes = b""
    cropped: bool = False

    @property
    def is_pdf(self) -> bool:
        return pdfs.is_pdf(self.original)


@dataclass
class AttachmentQueue:
    items: list[QueuedImage] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, name: str, data: bytes) -> tuple[bool, str]:
        """Validate and enqueue. Returns (added, message); message is empty on success."""
        if len(self.items) >= config.MAX_UPLOADS:
            return False, f"{name}: limit of {config.MAX_UPLOADS} screenshots per run reached."
        try:
            if pdfs.is_pdf(data):
                thumb = pdfs.thumbnail_png(data)
            else:
                load_image(data)
                thumb = thumbnail_png(data)
        except (ImageError, pdfs.PdfError) as exc:
            return False, f"{name}: {exc}."
        digest = image_hash(data)
        if any(i.hash == digest for i in self.items):
            return False, f"{name}: identical screenshot already attached."
        self.items.append(QueuedImage(next(_ids), name, data, data, digest, thumb))
        return True, ""

    def get(self, item_id: int) -> QueuedImage | None:
        return next((i for i in self.items if i.id == item_id), None)

    def remove(self, item_id: int) -> None:
        self.items = [i for i in self.items if i.id != item_id]

    def clear(self) -> None:
        self.items = []

    def move(self, item_id: int, delta: int) -> None:
        idx = next((n for n, i in enumerate(self.items) if i.id == item_id), None)
        if idx is None:
            return
        new = max(0, min(len(self.items) - 1, idx + delta))
        self.items.insert(new, self.items.pop(idx))

    def apply_crop(self, item_id: int, left: int, top: int, width: int, height: int) -> None:
        item = self.get(item_id)
        if item and not item.is_pdf:
            item.data = crop_percent(item.original, left, top, width, height)
            item.thumb = thumbnail_png(item.data)
            item.cropped = True

    def reset_crop(self, item_id: int) -> None:
        item = self.get(item_id)
        if item:
            item.data, item.cropped = item.original, False
            item.thumb = thumbnail_png(item.data)

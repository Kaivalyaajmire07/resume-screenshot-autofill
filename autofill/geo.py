"""Offline pincode -> city/state lookup (bundled India Post data, no network, no API).

Data: ``autofill/data/pincodes.csv`` (19,486 pincodes; derived from the open India Post
directory, MIT-licensed ``indiapins`` package). "City" is the India Post *district*, which is
the right answer for most cities (Pune, Nashik, Amravati) but can differ from a small town name.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data" / "pincodes.csv"
_PIN = re.compile(r"[1-9]\d{5}")


@dataclass(frozen=True)
class PinInfo:
    pincode: str
    city: str      # India Post district
    state: str
    office: str    # main post office / locality


@lru_cache(maxsize=1)
def _table() -> dict[str, PinInfo]:
    table: dict[str, PinInfo] = {}
    try:
        with _DATA.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                table[row["pincode"]] = PinInfo(row["pincode"], row["district"], row["state"], row["office"])
    except OSError:
        pass  # missing data file: lookups simply return None, the app keeps working
    return table


def lookup(pincode: str | None) -> PinInfo | None:
    pin = re.sub(r"\s", "", pincode or "")
    return _table().get(pin) if _PIN.fullmatch(pin) else None


def fill_location(values: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``values`` with blank city/state filled from the pincode. Never overwrites."""
    out = dict(values)
    info = lookup(out.get("pincode"))
    if info:
        if not (out.get("city") or "").strip():
            out["city"] = info.city
        if not (out.get("state") or "").strip():
            out["state"] = info.state
    return out

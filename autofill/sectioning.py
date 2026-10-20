"""Heading detection and section splitting shared by the extractors."""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_WORDS = (
    r"(?:career\s+)?objectives?|professional\s+summary|summary|profile(?:\s+summary)?|about\s+me|"
    r"(?:key|technical|core|professional)?\s*skills?(?:\s+summary)?|work\s+(?:experience|history)|"
    r"professional\s+experience|employment(?:\s+(?:history|details))?|experience|internships?(?:\s+experience)?|"
    r"education(?:al)?(?:\s+(?:qualifications?|details|background))?|academics?|"
    r"academic\s+(?:qualifications?|details|background|profile|achievements?)|qualifications?|projects?|"
    r"(?:academic\s+)?projects?|certifications?|(?:academic\s+)?achievements?|awards?|"
    r"personal\s+(?:details|information|profile)|languages?(?:\s+known)?|"
    r"hobbies|interests?|declaration|references?|contact(?:\s+(?:details|information))?|"
    r"strengths?|extra[\s-]?curricular(?:\s+activities)?|training"
)
HEADING_RE = re.compile(rf"^\W*(?:{HEADING_WORDS})\W*$", re.I)


def clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" \t•·▪●◦*-–—|")


def is_heading(line: str) -> bool:
    return len(line) < 45 and bool(HEADING_RE.match(line))


@dataclass
class Section:
    title: str  # lower-case heading text, "" for text before the first heading
    start: int  # index of the heading line (-1 for the leading block)
    body: list[int]  # indices of the lines belonging to the section


def split_sections(texts: list[str]) -> list[Section]:
    sections = [Section("", -1, [])]
    for i, text in enumerate(texts):
        if is_heading(text):
            sections.append(Section(re.sub(r"\W+", " ", text).strip().lower(), i, []))
        else:
            sections[-1].body.append(i)
    return [s for s in sections if s.body or s.title]

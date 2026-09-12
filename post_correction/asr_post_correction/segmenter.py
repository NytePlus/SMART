from __future__ import annotations

import re

from .models import Segment


BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])|(?<=[，,：:])")


def _split_long(text: str, absolute_start: int, max_chars: int) -> list[Segment]:
    if len(text) <= max_chars:
        return [Segment(0, absolute_start, absolute_start + len(text), text)]
    pieces: list[Segment] = []
    cursor = 0
    while cursor < len(text):
        proposed = min(cursor + max_chars, len(text))
        if proposed < len(text):
            window = text[cursor:proposed]
            breaks = [m.end() for m in re.finditer(r"\s+", window)]
            if breaks and breaks[-1] >= max_chars // 2:
                proposed = cursor + breaks[-1]
        piece = text[cursor:proposed]
        pieces.append(Segment(0, absolute_start + cursor, absolute_start + proposed, piece))
        cursor = proposed
    return pieces


def split_segments(hypothesis: str, max_chars: int = 80) -> list[Segment]:
    """Split without changing, dropping, or normalizing a single character."""
    if not hypothesis:
        return []
    segments: list[Segment] = []
    cursor = 0
    for match in BOUNDARY_RE.finditer(hypothesis):
        end = match.end()
        if end > cursor:
            segments.extend(_split_long(hypothesis[cursor:end], cursor, max_chars))
        cursor = end
    if cursor < len(hypothesis):
        segments.extend(_split_long(hypothesis[cursor:], cursor, max_chars))
    return [Segment(i, seg.start, seg.end, seg.text) for i, seg in enumerate(segments)]


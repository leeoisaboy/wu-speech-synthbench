from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PositionSource(str, Enum):
    READING = "reading"
    TEXT_FALLBACK = "text-fallback"
    NO_POSITION = "no-position"


@dataclass(frozen=True)
class DerivedPosition:
    source: PositionSource
    start_char: int | None
    end_char: int | None
    matched_phrase: str | None
    occurrence_index: int | None = None
    total_occurrences: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "char_source": self.source.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "matched_phrase": self.matched_phrase,
            "occurrence_index": self.occurrence_index,
            "total_occurrences": self.total_occurrences,
        }


def find_all_occurrences(reference_text: str, phrase: str) -> list[tuple[int, int]]:
    """Return all non-overlapping half-open character spans of ``phrase`` in ``reference_text``."""
    if not phrase:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    step = len(phrase)
    while True:
        idx = reference_text.find(phrase, cursor)
        if idx < 0:
            break
        spans.append((idx, idx + step))
        cursor = idx + step
    return spans


def _pick_occurrence(
    spans: list[tuple[int, int]],
    occurrence: int = 0,
    expected_char_center: float | None = None,
) -> tuple[int, int, int]:
    if expected_char_center is not None:
        best_idx = min(
            range(len(spans)),
            key=lambda i: abs(((spans[i][0] + spans[i][1]) / 2.0) - expected_char_center),
        )
        start, end = spans[best_idx]
        return start, end, best_idx
    if occurrence < 0 or occurrence >= len(spans):
        raise IndexError(f"occurrence index {occurrence} out of range for {len(spans)} matches")
    start, end = spans[occurrence]
    return start, end, occurrence


def derive_position(
    reference_text: str,
    predicted_phrase: str,
    fallback_phrase: str | None = None,
    occurrence: int = 0,
    expected_char_center: float | None = None,
) -> DerivedPosition:
    """Locate a prediction honestly; never synthesize proportional character offsets.

    When a phrase appears multiple times in a long utterance, ``expected_char_center``
    disambiguates by selecting the closest occurrence (otherwise ``occurrence`` picks
    the N-th match, defaulting to 0).
    """
    reading_spans = find_all_occurrences(reference_text, predicted_phrase)
    if reading_spans:
        start, end, occ_idx = _pick_occurrence(reading_spans, occurrence, expected_char_center)
        return DerivedPosition(
            source=PositionSource.READING,
            start_char=start,
            end_char=end,
            matched_phrase=predicted_phrase,
            occurrence_index=occ_idx,
            total_occurrences=len(reading_spans),
        )
    if fallback_phrase:
        fallback_spans = find_all_occurrences(reference_text, fallback_phrase)
        if fallback_spans:
            start, end, occ_idx = _pick_occurrence(fallback_spans, occurrence, expected_char_center)
            return DerivedPosition(
                source=PositionSource.TEXT_FALLBACK,
                start_char=start,
                end_char=end,
                matched_phrase=fallback_phrase,
                occurrence_index=occ_idx,
                total_occurrences=len(fallback_spans),
            )
    return DerivedPosition(PositionSource.NO_POSITION, None, None, None, None, 0)

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    segment_id: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Candidate:
    segment_id: int
    span: str
    start_in_segment: int
    end_in_segment: int
    confidence: float
    rubric: dict[str, str]
    reason: str
    evidence: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Suggestion:
    segment_id: int
    original: str
    replacement: str
    confidence: float
    rationale: str
    evidence: list[dict[str, str]] = field(default_factory=list)
    start_in_segment: int = -1
    end_in_segment: int = -1


@dataclass
class RoundResult:
    round_index: int
    input_hypothesis: str
    segments: list[Segment]
    candidates: list[Candidate]
    suggestions: list[Suggestion]
    verified_suggestions: list[Suggestion]
    verification_decisions: list[dict[str, Any]]
    output_hypothesis: str
    applied_edits: list[dict[str, Any]]
    rejected_edits: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectionResult:
    original_hypothesis: str
    corrected_hypothesis: str
    rounds: list[RoundResult]
    stop_reason: str
    seen_hypotheses: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

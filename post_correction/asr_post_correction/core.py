from __future__ import annotations

from dataclasses import asdict
import re
import unicodedata
from typing import Any, Protocol

from .models import Candidate, CorrectionResult, RoundResult, Segment, Suggestion
from .prompts import render
from .segmenter import split_segments


class JsonCompletionClient(Protocol):
    def complete_json(self, messages: list[dict[str, str]], *, purpose: str) -> tuple[dict[str, Any], dict[str, Any]]: ...


RUBRIC_KEYS = ("G1", "G2", "S1", "S2", "S3", "S4", "C1", "C2")
SPLIT_ASCII_LETTERS_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Za-z]\s+){1,}[A-Za-z](?=$|[^\sA-Za-z])"
)
INTERNAL_APOSTROPHE_RE = re.compile(r"(?<=[A-Za-z])[\u0027\u2019](?=[A-Za-z])")


class CorrectionEngine:
    """One-round and iterative correction with deterministic safety gates."""

    def __init__(
        self,
        client: JsonCompletionClient,
        prompt_bundle: dict[str, Any],
        *,
        detector_threshold: float = 0.72,
        corrector_threshold: float = 0.82,
        verifier_threshold: float = 0.82,
        max_segment_chars: int = 80,
        forbid_normalization_edits: bool = False,
    ):
        self.client = client
        self.prompts = prompt_bundle
        self.detector_threshold = detector_threshold
        self.corrector_threshold = corrector_threshold
        self.verifier_threshold = verifier_threshold
        self.max_segment_chars = max_segment_chars
        self.forbid_normalization_edits = forbid_normalization_edits
        self.deterministic_postprocessors = tuple(
            str(name) for name in prompt_bundle.get("deterministic_postprocessors", [])
        )

    @staticmethod
    def _messages(system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _annotated_alternate_disagreements(
        context: dict[str, Any], segments: list[Segment]
    ) -> list[dict[str, Any]]:
        """Attach primary segment IDs to non-GT alternate diff evidence."""
        annotated: list[dict[str, Any]] = []
        for source_item in context.get("alternate_disagreements", []) or []:
            if not isinstance(source_item, dict):
                continue
            source = str(source_item.get("source", ""))
            differences: list[dict[str, Any]] = []
            for item in source_item.get("differences", []) or []:
                if not isinstance(item, dict):
                    continue
                try:
                    start = int(item.get("hypothesis_char_start", 0))
                    end = int(item.get("hypothesis_char_end", start))
                except (TypeError, ValueError):
                    continue
                segment_ids = [
                    segment.segment_id
                    for segment in segments
                    if (start == end and segment.start <= start <= segment.end)
                    or (start < end and max(start, segment.start) < min(end, segment.end))
                ]
                differences.append({**item, "segment_ids": segment_ids})
            annotated.append({"source": source, "differences": differences})
        return annotated

    def detect(self, hypothesis: str, segments: list[Segment], context: dict[str, Any]) -> tuple[list[Candidate], dict[str, Any]]:
        payload = {
            "hypothesis": hypothesis,
            "segments": [asdict(segment) for segment in segments],
            "history": context.get("history", ""),
            "alternate_hypothesis": context.get("alternate_hypothesis", ""),
            "alternate_hypotheses": context.get("alternate_hypotheses", []),
            "alternate_disagreements": self._annotated_alternate_disagreements(context, segments),
            "slide_ocr": context.get("slide_ocr", ""),
            "metadata": context.get("metadata", {}),
        }
        passes = [("primary", self.prompts["detector_system"], self.prompts["detector_user"])]
        passes.extend(
            (str(item["name"]), str(item["system"]), str(item["user"]))
            for item in self.prompts.get("detector_passes", [])
        )
        candidates_by_span: dict[tuple[int, str, int, int], Candidate] = {}
        pass_meta: list[dict[str, Any]] = []
        by_id = {segment.segment_id: segment for segment in segments}
        for pass_name, system_prompt, user_prompt in passes:
            purpose = f"detect:{self.prompts['version']}"
            if pass_name != "primary":
                purpose += f":{pass_name}"
            raw, meta = self.client.complete_json(
                self._messages(system_prompt, render(user_prompt, payload)), purpose=purpose
            )
            pass_meta.append({"name": pass_name, **meta})
            for item in raw.get("candidates", []):
                try:
                    segment_id = int(item["segment_id"])
                    segment = by_id[segment_id]
                    span = str(item["span"])
                    start = int(item["start_in_segment"])
                    end = int(item["end_in_segment"])
                    confidence = float(item["confidence"])
                    rubric = dict(item["rubric"])
                except (KeyError, TypeError, ValueError):
                    continue
                if set(rubric) != set(RUBRIC_KEYS):
                    continue
                if not span or start < 0 or end <= start or segment.text[start:end] != span:
                    continue
                if confidence < self.detector_threshold:
                    continue
                if not any(rubric[key] == "可疑" for key in ("S1", "S2", "S3", "S4")):
                    continue
                candidate = Candidate(
                    segment_id=segment_id,
                    span=span,
                    start_in_segment=start,
                    end_in_segment=end,
                    confidence=confidence,
                    rubric=rubric,
                    reason=str(item.get("reason", "")),
                    evidence=list(item.get("evidence", [])),
                )
                key = (segment_id, span, start, end)
                existing = candidates_by_span.get(key)
                if existing is None:
                    candidates_by_span[key] = candidate
                    continue
                stronger = candidate if candidate.confidence > existing.confidence else existing
                reasons = list(dict.fromkeys(filter(None, (existing.reason, candidate.reason))))
                evidence = []
                seen_evidence: set[tuple[str, str]] = set()
                for entry in existing.evidence + candidate.evidence:
                    evidence_key = (str(entry.get("source", "")), str(entry.get("quote", "")))
                    if evidence_key not in seen_evidence:
                        seen_evidence.add(evidence_key)
                        evidence.append(entry)
                candidates_by_span[key] = Candidate(
                    segment_id=stronger.segment_id,
                    span=stronger.span,
                    start_in_segment=stronger.start_in_segment,
                    end_in_segment=stronger.end_in_segment,
                    confidence=stronger.confidence,
                    rubric=stronger.rubric,
                    reason="\n".join(reasons),
                    evidence=evidence,
                )
        candidates = list(candidates_by_span.values())
        candidates.sort(key=lambda x: (x.segment_id, x.start_in_segment, x.end_in_segment))
        meta = pass_meta[0] if len(pass_meta) == 1 else {"passes": pass_meta}
        return candidates, meta

    def suggest(self, hypothesis: str, segments: list[Segment], candidates: list[Candidate], context: dict[str, Any]) -> tuple[list[Suggestion], dict[str, Any]]:
        if not candidates:
            return [], {"skipped": "no_candidates"}
        payload = {
            "hypothesis": hypothesis,
            "segments": [asdict(segment) for segment in segments],
            "candidates": [asdict(candidate) for candidate in candidates],
            "history": context.get("history", ""),
            "alternate_hypothesis": context.get("alternate_hypothesis", ""),
            "alternate_hypotheses": context.get("alternate_hypotheses", []),
            "alternate_disagreements": self._annotated_alternate_disagreements(context, segments),
            "slide_ocr": context.get("slide_ocr", ""),
            "metadata": context.get("metadata", {}),
        }
        allowed = {(c.segment_id, c.span, c.start_in_segment, c.end_in_segment) for c in candidates}
        allowed_segment_ids = {c.segment_id for c in candidates}
        allow_subspan = bool(self.prompts.get("allow_subspan_suggestions"))
        segments_by_id = {segment.segment_id: segment for segment in segments}
        passes = [("primary", self.prompts["corrector_system"], self.prompts["corrector_user"])]
        passes.extend(
            (str(item["name"]), str(item["system"]), str(item["user"]))
            for item in self.prompts.get("corrector_passes", [])
        )
        suggestions_by_key: dict[tuple[int, str, int, int, str], Suggestion] = {}
        pass_meta: list[dict[str, Any]] = []
        for pass_name, system_prompt, user_prompt in passes:
            purpose = f"correct:{self.prompts['version']}"
            if pass_name != "primary":
                purpose += f":{pass_name}"
            raw, meta = self.client.complete_json(
                self._messages(system_prompt, render(user_prompt, payload)), purpose=purpose
            )
            pass_meta.append({"name": pass_name, **meta})
            for item in raw.get("suggestions", []):
                try:
                    suggestion = Suggestion(
                        segment_id=int(item["segment_id"]), original=str(item["original"]),
                        replacement=str(item["replacement"]), confidence=float(item["confidence"]),
                        rationale=str(item.get("rationale", "")), evidence=list(item.get("evidence", [])),
                        start_in_segment=int(item.get("start_in_segment", -1)),
                        end_in_segment=int(item.get("end_in_segment", -1)),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                exact_key = (suggestion.segment_id, suggestion.original,
                             suggestion.start_in_segment, suggestion.end_in_segment)
                legacy_match = any(c[0] == suggestion.segment_id and c[1] == suggestion.original for c in allowed)
                if allow_subspan:
                    segment = segments_by_id.get(suggestion.segment_id)
                    if (suggestion.segment_id not in allowed_segment_ids or segment is None
                            or suggestion.start_in_segment < 0
                            or segment.text[suggestion.start_in_segment:suggestion.end_in_segment] != suggestion.original):
                        continue
                elif exact_key not in allowed and not (suggestion.start_in_segment < 0 and legacy_match):
                    continue
                if not suggestion.replacement or suggestion.replacement == suggestion.original:
                    continue
                if suggestion.confidence < self.corrector_threshold:
                    continue
                if not self._evidence_is_extractive(suggestion.evidence, hypothesis, context):
                    continue
                key = (suggestion.segment_id, suggestion.original, suggestion.start_in_segment,
                       suggestion.end_in_segment, suggestion.replacement)
                previous = suggestions_by_key.get(key)
                if previous is None or suggestion.confidence > previous.confidence:
                    suggestions_by_key[key] = suggestion
        suggestions = list(suggestions_by_key.values())
        suggestions.sort(key=lambda x: (x.segment_id, x.start_in_segment, x.end_in_segment, x.replacement))
        meta = pass_meta[0] if len(pass_meta) == 1 else {"passes": pass_meta}
        return suggestions, meta

    def verify(self, hypothesis: str, segments: list[Segment], suggestions: list[Suggestion],
               context: dict[str, Any]) -> tuple[list[Suggestion], list[dict[str, Any]], dict[str, Any]]:
        """Independently accept proposed edits when a prompt bundle provides a verifier."""
        if not suggestions:
            return [], [], {"skipped": "no_suggestions"}
        if not self.prompts.get("verifier_system") or not self.prompts.get("verifier_user"):
            decisions = [{"suggestion_index": index, "accepted": True,
                          "reason": "no_verifier_configured"}
                         for index in range(len(suggestions))]
            return suggestions, decisions, {"skipped": "no_verifier_configured"}
        payload = {"hypothesis": hypothesis, "segments": [asdict(segment) for segment in segments],
                   "suggestions": [{"suggestion_index": index, **asdict(suggestion)}
                                   for index, suggestion in enumerate(suggestions)],
                   "history": context.get("history", ""), "metadata": context.get("metadata", {})}
        payload["alternate_hypothesis"] = context.get("alternate_hypothesis", "")
        payload["alternate_hypotheses"] = context.get("alternate_hypotheses", [])
        payload["alternate_disagreements"] = self._annotated_alternate_disagreements(context, segments)
        payload["slide_ocr"] = context.get("slide_ocr", "")
        raw, meta = self.client.complete_json(
            self._messages(self.prompts["verifier_system"], render(self.prompts["verifier_user"], payload)),
            purpose=f"verify:{self.prompts['version']}",
        )
        accepted: list[Suggestion] = []
        decisions: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for item in raw.get("decisions", []):
            try:
                index = int(item["suggestion_index"])
                accept = item["accept"] is True
                confidence = float(item["confidence"])
                evidence = list(item.get("evidence", []))
            except (KeyError, TypeError, ValueError):
                continue
            if index in seen_indices or index < 0 or index >= len(suggestions):
                continue
            seen_indices.add(index)
            evidence_valid = self._evidence_is_extractive(evidence, hypothesis, context)
            is_accepted = accept and confidence >= self.verifier_threshold and evidence_valid
            decisions.append({
                "suggestion_index": index,
                "model_accept": accept,
                "confidence": confidence,
                "accepted": is_accepted,
                "rationale": str(item.get("rationale", "")),
                "evidence": evidence,
                "evidence_valid": evidence_valid,
            })
            if is_accepted:
                accepted.append(suggestions[index])
        for index in range(len(suggestions)):
            if index not in seen_indices:
                decisions.append({
                    "suggestion_index": index,
                    "model_accept": False,
                    "accepted": False,
                    "reason": "missing_decision",
                })
        return accepted, decisions, meta

    @staticmethod
    def _evidence_is_extractive(evidence: list[dict[str, str]], hypothesis: str, context: dict[str, Any]) -> bool:
        if not evidence:
            return False
        sources = {
            "hypothesis": hypothesis,
            "history": str(context.get("history", "")),
            "alternate_hypothesis": str(context.get("alternate_hypothesis", "")),
            "slide_ocr": str(context.get("slide_ocr", "")),
        }
        for entry in evidence:
            source, quote = entry.get("source"), entry.get("quote")
            if source not in sources or not isinstance(quote, str) or not quote or quote not in sources[source]:
                return False
        return True

    def apply(self, hypothesis: str, segments: list[Segment], suggestions: list[Suggestion]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        by_id = {segment.segment_id: segment for segment in segments}
        accepted: list[tuple[int, int, Suggestion]] = []
        rejected: list[dict[str, Any]] = []
        for suggestion in suggestions:
            if self.forbid_normalization_edits and self._is_normalization_only_edit(
                suggestion.original, suggestion.replacement
            ):
                rejected.append({**asdict(suggestion), "reason": "normalization_only_edit"})
                continue
            segment = by_id.get(suggestion.segment_id)
            if segment is None:
                rejected.append({**asdict(suggestion), "reason": "unknown_segment"})
                continue
            if (suggestion.start_in_segment >= 0
                    and segment.text[suggestion.start_in_segment:suggestion.end_in_segment] == suggestion.original):
                positions = [suggestion.start_in_segment]
            else:
                positions = []
                cursor = 0
                while True:
                    found = segment.text.find(suggestion.original, cursor)
                    if found < 0:
                        break
                    positions.append(found)
                    cursor = found + 1
            if len(positions) != 1:
                rejected.append({**asdict(suggestion), "reason": "original_not_unique_in_segment"})
                continue
            start = segment.start + positions[0]
            accepted.append((start, start + len(suggestion.original), suggestion))
        accepted.sort(key=lambda x: x[0])
        non_overlapping: list[tuple[int, int, Suggestion]] = []
        previous_end = -1
        for edit in accepted:
            if edit[0] < previous_end:
                rejected.append({**asdict(edit[2]), "reason": "overlapping_edit"})
                continue
            non_overlapping.append(edit)
            previous_end = edit[1]
        output = hypothesis
        applied: list[dict[str, Any]] = []
        for start, end, suggestion in reversed(non_overlapping):
            if output[start:end] != suggestion.original:
                rejected.append({**asdict(suggestion), "reason": "stale_offset"})
                continue
            output = output[:start] + suggestion.replacement + output[end:]
            applied.append({**asdict(suggestion), "absolute_start": start, "absolute_end": end})
        applied.reverse()
        return output, applied, rejected

    @staticmethod
    def _is_normalization_only_edit(original: str, replacement: str) -> bool:
        """Identify case/spacing/punctuation-only changes without a dataset tokenizer."""
        def content_signature(text: str) -> str:
            normalized = unicodedata.normalize("NFKC", text).casefold()
            return "".join(character for character in normalized if character.isalnum())

        return original != replacement and content_signature(original) == content_signature(replacement)

    @staticmethod
    def _merge_split_ascii_letters(hypothesis: str) -> tuple[str, list[dict[str, Any]]]:
        matches = list(SPLIT_ASCII_LETTERS_RE.finditer(hypothesis))
        output = hypothesis
        edits: list[dict[str, Any]] = []
        for match in reversed(matches):
            original = match.group(0)
            replacement = re.sub(r"\s+", "", original)
            if replacement == original:
                continue
            start, end = match.start(), match.end()
            if output[start:end] != original:
                continue
            output = output[:start] + replacement + output[end:]
            edits.append({
                "original": original,
                "replacement": replacement,
                "absolute_start": start,
                "absolute_end": end,
                "source": "deterministic_postprocessor",
                "postprocessor": "merge_split_ascii_letters",
            })
        edits.reverse()
        return output, edits

    @staticmethod
    def _drop_chinese_disfluency_e(hypothesis: str) -> tuple[str, list[dict[str, Any]]]:
        edits: list[dict[str, Any]] = []
        output = hypothesis
        positions = [index for index, char in enumerate(hypothesis) if char == "呃"]
        for position in reversed(positions):
            if output[position:position + 1] != "呃":
                continue
            output = output[:position] + output[position + 1:]
            edits.append({
                "original": "呃",
                "replacement": "",
                "absolute_start": position,
                "absolute_end": position + 1,
                "source": "deterministic_postprocessor",
                "postprocessor": "drop_chinese_disfluency_e",
            })
        edits.reverse()
        return output, edits

    @staticmethod
    def _split_internal_apostrophes(hypothesis: str) -> tuple[str, list[dict[str, Any]]]:
        """Match benchmark orthography where an internal apostrophe is a boundary.

        TED-style references commonly encode forms such as ``I 'd`` and ``we 're``.
        Replacing only apostrophes surrounded by ASCII letters preserves possessive
        punctuation at a word edge and leaves all non-letter text unchanged.
        """
        matches = list(INTERNAL_APOSTROPHE_RE.finditer(hypothesis))
        output = hypothesis
        edits: list[dict[str, Any]] = []
        for match in reversed(matches):
            start, end = match.start(), match.end()
            original = match.group(0)
            if output[start:end] != original:
                continue
            output = output[:start] + " " + output[end:]
            edits.append({
                "original": original,
                "replacement": " ",
                "absolute_start": start,
                "absolute_end": end,
                "source": "deterministic_postprocessor",
                "postprocessor": "split_internal_apostrophes",
            })
        edits.reverse()
        return output, edits

    def apply_postprocessors(self, hypothesis: str) -> tuple[str, list[dict[str, Any]]]:
        output = hypothesis
        applied: list[dict[str, Any]] = []
        for name in self.deterministic_postprocessors:
            if name == "merge_split_ascii_letters":
                output, edits = self._merge_split_ascii_letters(output)
                applied.extend(edits)
            elif name == "drop_chinese_disfluency_e":
                output, edits = self._drop_chinese_disfluency_e(output)
                applied.extend(edits)
            elif name == "split_internal_apostrophes":
                output, edits = self._split_internal_apostrophes(output)
                applied.extend(edits)
        return output, applied

    def run_round(self, hypothesis: str, context: dict[str, Any], round_index: int = 1) -> RoundResult:
        segments = split_segments(hypothesis, self.max_segment_chars)
        candidates, detector_meta = self.detect(hypothesis, segments, context)
        suggestions, corrector_meta = self.suggest(hypothesis, segments, candidates, context)
        verified, verification_decisions, verifier_meta = self.verify(
            hypothesis, segments, suggestions, context
        )
        output, applied, rejected = self.apply(hypothesis, segments, verified)
        for edit in applied:
            edit["detector_meta"] = detector_meta
            edit["corrector_meta"] = corrector_meta
            edit["verifier_meta"] = verifier_meta
        output, postprocessed = self.apply_postprocessors(output)
        applied.extend(postprocessed)
        return RoundResult(round_index, hypothesis, segments, candidates, suggestions, verified,
                           verification_decisions, output, applied, rejected)

    def single_round(self, hypothesis: str, context: dict[str, Any] | None = None) -> CorrectionResult:
        result = self.run_round(hypothesis, context or {}, 1)
        stop = ("no_candidates" if not result.candidates and result.output_hypothesis == hypothesis
                else "no_changes" if result.output_hypothesis == hypothesis
                else "single_round_complete")
        return CorrectionResult(hypothesis, result.output_hypothesis, [result], stop, [hypothesis, result.output_hypothesis])

    def multi_round(self, hypothesis: str, context: dict[str, Any] | None = None, max_rounds: int = 4) -> CorrectionResult:
        context = context or {}
        current = hypothesis
        seen = [current]
        transitions: set[tuple[str, str]] = set()
        rounds: list[RoundResult] = []
        stop_reason = "max_rounds"
        for index in range(1, max_rounds + 1):
            result = self.run_round(current, context, index)
            rounds.append(result)
            proposed = result.output_hypothesis
            if not result.candidates:
                stop_reason = "no_candidates"
                break
            if proposed == current:
                stop_reason = "no_changes"
                break
            if (proposed, current) in transitions:
                stop_reason = "two_cycle_detected"
                break
            if proposed in seen:
                stop_reason = "repeated_state_detected"
                break
            transitions.add((current, proposed))
            current = proposed
            seen.append(current)
        return CorrectionResult(hypothesis, current, rounds, stop_reason, seen)

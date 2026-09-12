from __future__ import annotations

from difflib import SequenceMatcher
import unicodedata
from typing import Any, Sequence

from .io import _evidence_token_spans


def _content_signature(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _repeated_ngram(tokens: list[str], max_n: int) -> bool:
    for size in range(1, min(max_n, len(tokens) // 2) + 1):
        for index in range(len(tokens) - 2 * size + 1):
            if tokens[index:index + size] == tokens[index + size:index + 2 * size]:
                return True
    return False


def _repeat_touches_edit(
    tokens: list[str], boundary: int, replacement_size: int, max_n: int
) -> bool:
    """Detect a new adjacent repeated unit that crosses the edited span."""
    for size in range(1, min(max_n, len(tokens) // 2) + 1):
        for index in range(len(tokens) - 2 * size + 1):
            if tokens[index:index + size] != tokens[index + size:index + 2 * size]:
                continue
            if index <= boundary + replacement_size and index + 2 * size >= boundary:
                return True
    return False


def _token_span_conflicts(
    left_start: int, left_end: int, right_start: int, right_end: int
) -> bool:
    """Treat two insertions at one boundary as conflicting edits."""
    if left_start == left_end and right_start == right_end:
        return left_start == right_start
    if left_start == left_end:
        return right_start <= left_start < right_end
    if right_start == right_end:
        return left_start <= right_start < left_end
    return left_start < right_end and right_start < left_end


def _token_length_imbalance(
    original_tokens: Sequence[str],
    replacement_tokens: Sequence[str],
    limit: float,
) -> bool:
    """Reject replacements whose token length changes are implausibly large.

    A single alternate can align a long region to one token when it has lost
    most of the utterance.  Such an edit is not a local correction signal even
    though its character span may fit the ordinary span limit.  One- or
    two-token insertions/deletions remain allowed for common function-word and
    disfluency differences.
    """
    original_size = len(original_tokens)
    replacement_size = len(replacement_tokens)
    if original_size == 0 or replacement_size == 0:
        return max(original_size, replacement_size) > 2
    return max(original_size, replacement_size) / min(original_size, replacement_size) > limit


def propose_local_edits(
    primary: str,
    alternate: str,
    *,
    max_ngram: int = 4,
    max_span_chars: int = 15,
    max_token_ratio: float = 4.0,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply conservative, content-only local differences from one alternate.

    This is an experimental candidate generator, not an alternate transcript
    replacement policy. It has no access to reference text. Pure formatting
    differences and edits that introduce a new repeated local n-gram are
    rejected; callers should still use an independent semantic verifier.
    """
    primary_spans = _evidence_token_spans(primary)
    alternate_spans = _evidence_token_spans(alternate)
    primary_tokens = [item[0] for item in primary_spans]
    alternate_tokens = [item[0] for item in alternate_spans]
    operations: list[tuple[int, int, str, str]] = []

    matcher = SequenceMatcher(a=primary_tokens, b=alternate_tokens, autojunk=False)
    for tag, primary_start, primary_end, alternate_start, alternate_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if primary_start < primary_end:
            start = primary_spans[primary_start][1]
            end = primary_spans[primary_end - 1][2]
        else:
            start = end = (
                primary_spans[primary_start][1]
                if primary_start < len(primary_spans)
                else len(primary)
            )
        replacement = (
            alternate[alternate_spans[alternate_start][1]:alternate_spans[alternate_end - 1][2]]
            if alternate_start < alternate_end
            else ""
        )
        original = primary[start:end]
        if not replacement and not original:
            continue
        if original == replacement or _content_signature(original) == _content_signature(replacement):
            continue
        if len(original) > max_span_chars or len(replacement) > max_span_chars:
            continue

        left = primary_tokens[max(0, primary_start - 4):primary_start]
        right = primary_tokens[primary_end:primary_end + 4]
        replacement_tokens = alternate_tokens[alternate_start:alternate_end]
        if _token_length_imbalance(
            primary_tokens[primary_start:primary_end], replacement_tokens, max_token_ratio
        ):
            continue
        if _repeated_ngram(replacement_tokens, max_ngram):
            continue
        local_after = left + replacement_tokens + right
        if _repeat_touches_edit(local_after, len(left), len(replacement_tokens), max_ngram):
            continue
        operations.append((start, end, replacement, tag))

    output = primary
    edits: list[dict[str, Any]] = []
    for start, end, replacement, operation in reversed(operations):
        original = output[start:end]
        output = output[:start] + replacement + output[end:]
        edits.append({
            "original": original,
            "replacement": replacement,
            "absolute_start": start,
            "absolute_end": end,
            "source": "alternate_local_selector",
            "operation": operation,
            "safety_gate": "span_and_token_ratio_and_new_repeated_ngram_rejected",
        })
    edits.reverse()
    return output, edits


def propose_consensus_edits(
    primary: str,
    alternates: Sequence[str],
    *,
    min_agreement: int = 2,
    max_ngram: int = 4,
    max_span_chars: int = 15,
    max_token_ratio: float = 4.0,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply only content edits independently supported by multiple alternates.

    Each alternate is aligned to the primary independently.  An edit is
    eligible only when distinct alternates identify the same primary token span
    and the same replacement content.  This is intentionally opt-in: agreement
    is evidence for a verifier, not an assertion that an alternate is truth.
    """
    primary_spans = _evidence_token_spans(primary)
    primary_tokens = [item[0] for item in primary_spans]
    proposals: dict[tuple[int, int, str], list[int]] = {}

    for source_index, alternate in enumerate(alternates):
        if not alternate:
            continue
        alternate_spans = _evidence_token_spans(alternate)
        alternate_tokens = [item[0] for item in alternate_spans]
        matcher = SequenceMatcher(a=primary_tokens, b=alternate_tokens, autojunk=False)
        seen_keys: set[tuple[int, int, str]] = set()
        for tag, primary_start, primary_end, alternate_start, alternate_end in matcher.get_opcodes():
            if tag == "equal":
                continue
            if primary_start < primary_end:
                start = primary_spans[primary_start][1]
                end = primary_spans[primary_end - 1][2]
            else:
                start = end = (
                    primary_spans[primary_start][1]
                    if primary_start < len(primary_spans)
                    else len(primary)
                )
            replacement = (
                alternate[alternate_spans[alternate_start][1]:alternate_spans[alternate_end - 1][2]]
                if alternate_start < alternate_end
                else ""
            )
            original = primary[start:end]
            if not replacement and not original:
                continue
            if original == replacement or _content_signature(original) == _content_signature(replacement):
                continue
            if len(original) > max_span_chars or len(replacement) > max_span_chars:
                continue
            replacement_tokens = alternate_tokens[alternate_start:alternate_end]
            if _token_length_imbalance(
                primary_tokens[primary_start:primary_end], replacement_tokens, max_token_ratio
            ):
                continue
            left = primary_tokens[max(0, primary_start - 4):primary_start]
            right = primary_tokens[primary_end:primary_end + 4]
            if _repeated_ngram(replacement_tokens, max_ngram):
                continue
            local_after = left + replacement_tokens + right
            if _repeat_touches_edit(local_after, len(left), len(replacement_tokens), max_ngram):
                continue
            key = (primary_start, primary_end, _content_signature(replacement))
            if key not in seen_keys:
                proposals.setdefault(key, []).append(source_index)
                seen_keys.add(key)

    eligible = [
        (primary_start, primary_end, replacement_signature, sources)
        for (primary_start, primary_end, replacement_signature), sources in proposals.items()
        if len(sources) >= min_agreement
    ]
    # Prefer stronger agreement, then larger spans, and resolve overlaps
    # deterministically before applying edits from right to left.
    eligible.sort(key=lambda item: (len(item[3]), item[1] - item[0]), reverse=True)
    selected: list[tuple[int, int, str, list[int]]] = []
    occupied: list[tuple[int, int]] = []
    for item in eligible:
        primary_start, primary_end, _, _ = item
        if any(
            _token_span_conflicts(primary_start, primary_end, start, end)
            for start, end in occupied
        ):
            continue
        selected.append(item)
        occupied.append((primary_start, primary_end))

    # Recover replacement text from the first alternate that contributed to the
    # selected proposal.  The signature gate above makes source disagreement on
    # content impossible for an eligible key.
    operations: list[tuple[int, int, str, list[int]]] = []
    for primary_start, primary_end, replacement_signature, sources in selected:
        replacement = ""
        recovered = False
        for source_index in sources:
            alternate_spans = _evidence_token_spans(alternates[source_index])
            alternate_tokens = [item[0] for item in alternate_spans]
            matcher = SequenceMatcher(a=primary_tokens, b=alternate_tokens, autojunk=False)
            for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
                if tag != "equal" and a_start == primary_start and a_end == primary_end:
                    replacement = alternates[source_index][
                        alternate_spans[b_start][1]:alternate_spans[b_end - 1][2]
                    ] if b_start < b_end else ""
                    if _content_signature(replacement) == replacement_signature:
                        recovered = True
                        break
            if recovered:
                break
        if recovered:
            if primary_start < primary_end:
                operation_start = primary_spans[primary_start][1]
                operation_end = primary_spans[primary_end - 1][2]
            else:
                operation_start = operation_end = (
                    primary_spans[primary_start][1]
                    if primary_start < len(primary_spans)
                    else len(primary)
                )
            operations.append((
                operation_start,
                operation_end,
                replacement,
                sources,
            ))

    output = primary
    edits: list[dict[str, Any]] = []
    for start, end, replacement, sources in sorted(
        operations, key=lambda item: (item[0], item[1]), reverse=True
    ):
        original = output[start:end]
        output = output[:start] + replacement + output[end:]
        edits.append({
            "original": original,
            "replacement": replacement,
            "absolute_start": start,
            "absolute_end": end,
            "source": "alternate_consensus_selector",
            "supporting_alternates": len(sources),
            "safety_gate": "multi_alternate_content_agreement_and_span_token_ratio_gate",
        })
    edits.reverse()
    return output, edits

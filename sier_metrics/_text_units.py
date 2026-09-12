"""Shared jieba chunk and word counting for benchmark metrics."""

from __future__ import annotations

import re
import unicodedata

import jieba

from asr_testing.wer import tokenize_asr_text

# Align CJK detection with text_normalization/spacing.py.
_CJK_CHAR_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")


def _is_punctuation_or_symbol(token: str) -> bool:
    return all(unicodedata.category(ch).startswith(("P", "S")) for ch in token)


def iter_jieba_chunks(text: str) -> list[str]:
    """Return effective jieba chunks (punctuation/symbol tokens filtered out)."""
    chunks: list[str] = []
    for token in jieba.lcut(str(text or "")):
        stripped = token.strip()
        if not stripped or _is_punctuation_or_symbol(stripped):
            continue
        chunks.append(stripped)
    return chunks


def text_chunk_count(text: str) -> int:
    return len(iter_jieba_chunks(text))


def word_count(chunk: str) -> int:
    """Count WER-compatible tokens: CJK by character, English by word."""
    return len(tokenize_asr_text(str(chunk or "")))


def text_word_count(text: str) -> int:
    """Count target words with the same tokenizer used by WER."""
    return len(tokenize_asr_text(str(text or "")))


def iter_word_spans(text: str) -> list[tuple[int, int]]:
    """Return raw-text spans using the same unit policy as WER tokenization."""
    spans: list[tuple[int, int]] = []
    index = 0
    raw = unicodedata.normalize("NFKC", str(text or ""))
    while index < len(raw):
        char = raw[index]
        if char.isspace() or _is_punctuation_or_symbol(char):
            index += 1
            continue

        start = index
        if char.isascii() and char.isalnum():
            index += 1
            while index < len(raw) and raw[index].isascii() and raw[index].isalnum():
                index += 1
        else:
            index += 1
        spans.append((start, index))
    return spans


def build_ref_error_flags(reference: str, op_segment: str) -> list[bool]:
    """
    Mark reference characters aligned to substitution/insertion ops as errors.

    Ops follow segment_mapping_core backtracking symbols:
      '=' match, '$' substitution, '+' deletion from hypothesis, '-' insertion in hypothesis.
    """
    ref = str(reference or "")
    if not ref:
        return []

    ops = str(op_segment or "")
    if not ops:
        return [True] * len(ref)

    flags = [False] * len(ref)
    ref_pos = 0
    for op in ops:
        if op == "=":
            if ref_pos < len(ref):
                flags[ref_pos] = False
                ref_pos += 1
        elif op in ("$", "+"):
            if ref_pos < len(ref):
                flags[ref_pos] = True
                ref_pos += 1
        elif op == "-":
            continue
        else:
            raise ValueError(f"unknown op_segment symbol: {op!r}")

    if ref_pos != len(ref):
        raise ValueError(
            f"op_segment/reference length mismatch: ref={len(ref)!r} consumed={ref_pos!r} ops={ops!r}"
        )
    return flags


def build_hyp_error_flags(hypothesis: str, op_segment: str) -> list[bool]:
    """Mark hypothesis characters affected by substitutions, insertions, or deletions.

    A deletion (``+``) consumes no hypothesis character. Attach it to the
    character immediately to the left of the deletion boundary; for a leading
    deletion, attach it to the first hypothesis character. This gives every
    deletion a hypothesis-side anchor for c-SEER2.
    """
    hyp = str(hypothesis or "")
    if not hyp:
        return []

    ops = str(op_segment or "")
    if not ops:
        return [True] * len(hyp)

    flags = [False] * len(hyp)
    hyp_pos = 0
    for op in ops:
        if op == "=":
            if hyp_pos < len(hyp):
                hyp_pos += 1
        elif op in ("$", "-"):
            if hyp_pos < len(hyp):
                flags[hyp_pos] = True
                hyp_pos += 1
        elif op == "+":
            anchor = hyp_pos - 1 if hyp_pos > 0 else 0
            if anchor < len(hyp):
                flags[anchor] = True
        else:
            raise ValueError(f"unknown op_segment symbol: {op!r}")

    if hyp_pos != len(hyp):
        raise ValueError(
            f"op_segment/hypothesis length mismatch: hyp={len(hyp)!r} consumed={hyp_pos!r} ops={ops!r}"
        )
    return flags


def error_word_count_in_reference(reference: str, op_segment: str) -> int:
    """Count minimum word units in reference that contain at least one error char."""
    ref = str(reference or "")
    if not ref:
        return 0
    flags = build_ref_error_flags(ref, op_segment)
    return sum(1 for start, end in iter_word_spans(ref) if any(flags[start:end]))


def error_chunk_count_in_reference(reference: str, op_segment: str) -> int:
    """Count jieba chunks in reference that contain at least one error word."""
    ref = str(reference or "")
    if not ref:
        return 0
    flags = build_ref_error_flags(ref, op_segment)
    search_from = 0
    error_chunks = 0
    for chunk in iter_jieba_chunks(ref):
        idx = ref.find(chunk, search_from)
        if idx < 0:
            idx = ref.find(chunk)
        if idx < 0:
            if any(flags):
                error_chunks += 1
            continue
        end = idx + len(chunk)
        search_from = end
        if any(flags[idx:end]):
            error_chunks += 1
    return error_chunks


def error_chunk_count_in_hypothesis(hypothesis: str, op_segment: str) -> int:
    """Count hyp chunks containing an error or anchoring a ref-side deletion."""
    hyp = str(hypothesis or "")
    if not hyp:
        return 0
    flags = build_hyp_error_flags(hyp, op_segment)
    search_from = 0
    chunk_spans: list[tuple[int, int]] = []
    for chunk in iter_jieba_chunks(hyp):
        idx = hyp.find(chunk, search_from)
        if idx < 0:
            idx = hyp.find(chunk)
        if idx < 0:
            continue
        end = idx + len(chunk)
        search_from = end
        chunk_spans.append((idx, end))

    # Re-anchor every deletion to the nearest real hypothesis chunk. The raw
    # character anchor may be whitespace or punctuation, which jieba omits.
    hyp_pos = 0
    for op in str(op_segment or ""):
        if op in ("=", "$", "-"):
            hyp_pos += 1
            continue
        if op != "+" or not chunk_spans:
            continue

        containing = next(
            ((start, end) for start, end in chunk_spans if start < hyp_pos < end),
            None,
        )
        left = [span for span in chunk_spans if span[1] <= hyp_pos]
        right = [span for span in chunk_spans if span[0] >= hyp_pos]
        if containing is not None:
            anchor = hyp_pos - 1
        elif left:
            anchor = left[-1][1] - 1
        elif right:
            anchor = right[0][0]
        else:
            anchor = chunk_spans[0][0]
        flags[anchor] = True

    if not chunk_spans:
        # A punctuation-only hypothesis has no jieba chunk. Preserve the error
        # as one fallback chunk instead of losing the deletion entirely.
        return int(any(flags))
    return sum(1 for start, end in chunk_spans if any(flags[start:end]))


def reference_word_count(reference: str) -> int:
    return len(iter_word_spans(str(reference or "")))


def non_error_word_count_in_reference(reference: str, op_segment: str) -> int:
    ref = str(reference or "")
    if not ref:
        return 0
    return reference_word_count(ref) - error_word_count_in_reference(ref, op_segment)


def non_error_chunk_count_in_reference(reference: str, op_segment: str) -> int:
    ref = str(reference or "")
    if not ref:
        return 0
    return text_chunk_count(ref) - error_chunk_count_in_reference(ref, op_segment)


def segment_error_unit_counts(reference: str, op_segment: str) -> dict[str, int]:
    ref = str(reference or "")
    op = str(op_segment or "")
    error_chunks = error_chunk_count_in_reference(ref, op)
    error_words = error_word_count_in_reference(ref, op)
    return {
        "error_chunks": error_chunks,
        "non_error_chunks": non_error_chunk_count_in_reference(ref, op),
        "error_words": error_words,
        "non_error_words": non_error_word_count_in_reference(ref, op),
    }

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


EVIDENCE_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


TAG_RE = re.compile(r"<\|[^|<>]+\|>")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"bad JSON at {path}:{number}: {exc}") from exc
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_text(row: dict[str, Any]) -> str:
    value = row.get("pred", row.get("prediction", ""))
    return TAG_RE.sub("", str(value)).strip()


def _evidence_tokens(text: str) -> list[str]:
    """Tokenize only for non-GT disagreement evidence shown to the LLM."""
    return [match.group(0).casefold() for match in EVIDENCE_TOKEN_RE.finditer(text)]


def _evidence_token_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).casefold(), match.start(), match.end())
        for match in EVIDENCE_TOKEN_RE.finditer(text)
    ]


def _token_char_bounds(
    spans: list[tuple[str, int, int]], start: int, end: int, text_length: int
) -> tuple[int, int]:
    if start < end:
        return spans[start][1], spans[end - 1][2]
    point = spans[start][1] if start < len(spans) else text_length
    return point, point


def _disagreement_summary(primary: str, alternate: str, *, limit: int = 32) -> list[dict[str, Any]]:
    """Return bounded local token disagreements between two hypotheses.

    This is evidence construction, not a correction rule: it never selects a
    replacement and it has no access to target text.
    """
    primary_spans = _evidence_token_spans(primary)
    alternate_spans = _evidence_token_spans(alternate)
    primary_tokens = [item[0] for item in primary_spans]
    alternate_tokens = [item[0] for item in alternate_spans]
    matcher = SequenceMatcher(a=primary_tokens, b=alternate_tokens, autojunk=False)
    differences: list[dict[str, Any]] = []
    for tag, p_start, p_end, a_start, a_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        primary_context_start = max(0, p_start - 4)
        primary_context_end = min(len(primary_tokens), p_end + 4)
        alternate_context_start = max(0, a_start - 4)
        alternate_context_end = min(len(alternate_tokens), a_end + 4)
        primary_start, primary_end = _token_char_bounds(
            primary_spans, p_start, p_end, len(primary)
        )
        alternate_start, alternate_end = _token_char_bounds(
            alternate_spans, a_start, a_end, len(alternate)
        )
        primary_context_char_start, primary_context_char_end = _token_char_bounds(
            primary_spans, primary_context_start, primary_context_end, len(primary)
        )
        alternate_context_char_start, alternate_context_char_end = _token_char_bounds(
            alternate_spans, alternate_context_start, alternate_context_end, len(alternate)
        )
        differences.append({
            "operation": tag,
            "hypothesis_token_start": p_start,
            "hypothesis_token_end": p_end,
            "alternate_token_start": a_start,
            "alternate_token_end": a_end,
            "hypothesis_tokens": primary_tokens[p_start:p_end],
            "alternate_tokens": alternate_tokens[a_start:a_end],
            "hypothesis_context": primary_tokens[primary_context_start:primary_context_end],
            "alternate_context": alternate_tokens[alternate_context_start:alternate_context_end],
            "hypothesis_char_start": primary_start,
            "hypothesis_char_end": primary_end,
            "alternate_char_start": alternate_start,
            "alternate_char_end": alternate_end,
            "hypothesis_context_text": primary[
                primary_context_char_start:primary_context_char_end
            ],
            "alternate_context_text": alternate[
                alternate_context_char_start:alternate_context_char_end
            ],
        })
        if len(differences) >= limit:
            break
    return differences


def target_text(row: dict[str, Any]) -> str:
    return str(row.get("target", row.get("ground_truth", "")))


def session_key(row: dict[str, Any]) -> str:
    return str(row.get("session_id") or row.get("session") or row.get("talk_id") or "__global__")


def source_id(row: dict[str, Any]) -> str:
    """Return the dataset-local ID shared by converted and native predictions."""
    if row.get("source_id") is not None:
        return str(row["source_id"])
    return str(row.get("id", "")).rsplit(":", 1)[-1]


def build_contexts(predictions: list[dict[str, Any]], message_rows: Iterable[dict[str, Any]], history_size: int) -> dict[str, dict[str, Any]]:
    metadata = {str(row.get("id")): row for row in message_rows}
    histories: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=history_size))
    contexts: dict[str, dict[str, Any]] = {}
    for row in predictions:
        row_id = str(row.get("id"))
        meta = metadata.get(source_id(row), metadata.get(row_id, {}))
        key = session_key(meta)
        contexts[row_id] = {"history": "\n".join(histories[key]), "metadata": {
            k: meta[k] for k in ("area", "subarea", "speaker", "session", "talk_id") if k in meta
        }}
        text = prediction_text(row)
        if text:
            histories[key].append(text)
    return contexts


def add_auxiliary_context(
    contexts: dict[str, dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    *,
    alternate_rows: Iterable[dict[str, Any]] = (),
    ocr_rows: Iterable[dict[str, Any]] = (),
) -> None:
    """Attach non-GT alternate transcripts and slide OCR by dataset-local ID."""
    alternate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in alternate_rows:
        text = prediction_text(row)
        values = alternate[source_id(row)]
        source = str(
            row.get("model")
            or row.get("display_name")
            or row.get("source")
            or row.get("_evidence_source")
            or f"ASR{len(values) + 1}"
        )
        if text and not any(
            item["source"] == source and item["text"] == text for item in values
        ):
            values.append({"source": source, "text": text})
    ocr: dict[str, str] = {}
    for row in ocr_rows:
        value = row.get("ocr", {})
        text = value.get("text", "") if isinstance(value, dict) else ""
        if text:
            ocr[source_id(row)] = str(text)
    for row in predictions:
        row_id = str(row.get("id"))
        context = contexts.get(row_id)
        if context is None:
            continue
        local_id = source_id(row)
        if alternate.get(local_id):
            values = alternate[local_id]
            labeled = [
                f"{item['source']}: {item['text']}"
                if item["source"].startswith("ASR")
                else f"[{item['source']}] {item['text']}"
                for item in values
            ]
            context["alternate_hypothesis"] = (
                values[0]["text"]
                if len(values) == 1 and values[0]["source"].startswith("ASR")
                else labeled[0] if len(labeled) == 1 else "\n\n".join(labeled)
            )
            context["alternate_hypotheses"] = [dict(item) for item in values]
            context["alternate_disagreements"] = [
                {
                    "source": item["source"],
                    "differences": _disagreement_summary(
                        prediction_text(row), item["text"]
                    ),
                }
                for item in values
            ]
        if ocr.get(local_id):
            context["slide_ocr"] = ocr[local_id]


def load_completed_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    return {str(row.get("id")) for row in read_jsonl(output) if row.get("status") == "ok"}


def systematic_indices(total: int, count: int) -> list[int]:
    """Choose positions without inspecting record content."""
    if count <= 0 or count > total:
        raise ValueError(f"count must be between 1 and {total}")
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]

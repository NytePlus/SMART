# -*- coding: utf-8 -*-
"""
Character/word mapping based on edit-distance alignment and jieba segmentation.

This module provides a lightweight mapping between a reference text and a
hypothesis/prediction text. It uses classic Levenshtein edit-distance to
produce character-level alignments, then groups characters into word-level
segments using jieba word boundaries (and spaces for English).

Public API:
    compute_mapping(reference, hypothesis)
"""

import sys
from pathlib import Path

# Allow importing jieba from the project-local virtual environment.
_VENV_SITE_PACKAGES = (
    Path(__file__).parent.parent
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if _VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(_VENV_SITE_PACKAGES))

try:
    import jieba

    # Use a project-local cache directory to avoid /tmp permission issues.
    _JIEBA_CACHE_DIR = Path(__file__).parent.parent / ".jieba_cache"
    _JIEBA_CACHE_DIR.mkdir(exist_ok=True)
    jieba.dt.tmp_dir = str(_JIEBA_CACHE_DIR)
except ImportError:
    jieba = None


# ---------------------------------------------------------------------------
# Word boundary helpers (jieba)
# ---------------------------------------------------------------------------

def _get_jieba_boundaries(text):
    """
    Return the set of word boundary positions using jieba.

    A position p is a boundary if it is the start of a non-space jieba token
    other than the first token.
    """
    if jieba is None:
        raise ImportError(
            "jieba is not installed. Please run:\n"
            "  python3 -m venv .venv\n"
            "  .venv/bin/pip install jieba"
        )
    boundaries = set()
    tokens = list(jieba.tokenize(text))
    for i, (word, start, _end) in enumerate(tokens):
        if i == 0:
            continue
        if word != " ":
            boundaries.add(start)
    return boundaries


# ---------------------------------------------------------------------------
# Edit-distance alignment
# ---------------------------------------------------------------------------

def _edit_distance_dp(s1, s2):
    """Compute edit-distance DP matrix between s1 and s2."""
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j - 1], dp[i - 1][j])
    return dp


def _backtrack_symbols(s1, s2, dp):
    """
    Backtrack the DP matrix and return the operation sequence.

    Here s1 is the hypothesis and s2 is the reference.  Operations:
        '='  match
        '$'  substitution
        '+'  insertion in hypothesis (missing char in hypothesis)
        '-'  deletion from hypothesis (extra char in hypothesis)
    """
    ops = []
    i, j = len(s1), len(s2)
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            ops.append("=")
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append("$")
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            ops.append("-")
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            ops.append("+")
            j -= 1
    while i > 0:
        ops.append("-")
        i -= 1
    while j > 0:
        ops.append("+")
        j -= 1
    return ops[::-1]


def _build_columns(reference, hypothesis, ops):
    """
    Convert the operation sequence into explicit alignment columns.

    Returns:
        columns: list of (h_char, op, gt_char), where h_char or gt_char may
                 be None to indicate a gap.
    """
    columns = []
    hi, gi = 0, 0
    for op in ops:
        if op == "=":
            columns.append((hypothesis[hi], "=", reference[gi]))
            hi += 1
            gi += 1
        elif op == "$":
            columns.append((hypothesis[hi], "$", reference[gi]))
            hi += 1
            gi += 1
        elif op == "-":
            # Hypothesis has an extra char, reference has a gap.
            columns.append((hypothesis[hi], "-", None))
            hi += 1
        elif op == "+":
            # Hypothesis misses a char, reference has the char.
            columns.append((None, "+", reference[gi]))
            gi += 1
    return columns


# ---------------------------------------------------------------------------
# Word-level mapping
# ---------------------------------------------------------------------------

def _map_to_word_segments(reference, hypothesis, columns):
    """
    Group alignment columns into word-level segments.

    Segment boundaries are placed at:
      1. Aligned spaces (preserves original SemaScore spacing-error grouping).
      2. Common non-space word boundaries, where both the reference and the
         hypothesis have a jieba word boundary.
    """
    ref_boundaries = _get_jieba_boundaries(reference)
    hyp_boundaries = _get_jieba_boundaries(hypothesis)

    ref_segments, hyp_segments, op_segments = [], [], []
    temp_ref, temp_hyp, temp_op = "", "", ""
    ref_idx, hyp_idx = 0, 0

    for h_char, op, gt_char in columns:
        # Aligned spaces act as separators.
        if gt_char == " " and h_char == " ":
            ref_segments.append(temp_ref)
            hyp_segments.append(temp_hyp)
            op_segments.append(temp_op)
            temp_ref, temp_hyp, temp_op = "", "", ""
            ref_idx += 1
            hyp_idx += 1
            continue

        # Common non-space word boundary.
        at_ref_boundary = ref_idx > 0 and ref_idx in ref_boundaries
        at_hyp_boundary = hyp_idx > 0 and hyp_idx in hyp_boundaries
        if at_ref_boundary and at_hyp_boundary:
            ref_segments.append(temp_ref)
            hyp_segments.append(temp_hyp)
            op_segments.append(temp_op)
            temp_ref, temp_hyp, temp_op = "", "", ""

        if gt_char is not None:
            temp_ref += gt_char
            ref_idx += 1
        if h_char is not None:
            temp_hyp += h_char
            hyp_idx += 1
        temp_op += op

    ref_segments.append(temp_ref)
    hyp_segments.append(temp_hyp)
    op_segments.append(temp_op)

    # Filter out segments that are empty in all three lists.
    filtered = [
        (r, h, o)
        for r, h, o in zip(ref_segments, hyp_segments, op_segments)
        if r or h or o
    ]
    if not filtered:
        return [], [], []
    mapped_reference, mapped_hypothesis, mapped_ops = zip(*filtered)
    return _merge_empty_segments(
        list(mapped_reference), list(mapped_hypothesis), list(mapped_ops)
    )


def _merge_empty_segments(ref_segs, hyp_segs, op_segs):
    """
    将插入/删除产生的空片段向前合并，避免单个插入/删除字被孤立成独立 segment。

    规则：
      - 若某个 segment 的 reference 或 hypothesis 任意一侧为空（插入或删除），
        则把它向前合并到下一个 segment；若已是最后一个，则向后合并。
      - 匹配/替换 segment（两侧均非空）保持独立。
    """
    if not ref_segs:
        return ref_segs, hyp_segs, op_segs

    def _is_empty_side(i):
        return not ref_segs[i] or not hyp_segs[i]

    i = 0
    while i < len(ref_segs):
        if not _is_empty_side(i):
            i += 1
            continue

        # 当前 segment 是插入/删除，优先向前合并
        if i + 1 < len(ref_segs):
            ref_segs[i + 1] = ref_segs[i] + ref_segs[i + 1]
            hyp_segs[i + 1] = hyp_segs[i] + hyp_segs[i + 1]
            op_segs[i + 1] = op_segs[i] + op_segs[i + 1]
            del ref_segs[i], hyp_segs[i], op_segs[i]
            # 继续检查合并后的位置，因为它仍可能是空片段
            continue
        elif i > 0:
            # 最后一个 segment，向后合并
            ref_segs[i - 1] = ref_segs[i - 1] + ref_segs[i]
            hyp_segs[i - 1] = hyp_segs[i - 1] + hyp_segs[i]
            op_segs[i - 1] = op_segs[i - 1] + op_segs[i]
            del ref_segs[i], hyp_segs[i], op_segs[i]
            # 删除后继续，无需递增
            continue
        else:
            # 只有一个 segment 且为空侧，无法合并
            i += 1

    return ref_segs, hyp_segs, op_segs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_mapping(reference, hypothesis):
    """
    Compute edit-distance-based mapping between reference and hypothesis.

    Args:
        reference: ground-truth / labels text
        hypothesis: prediction / hypothesis text

    Returns:
        dict with keys:
            - reference: original reference
            - hypothesis: original hypothesis
            - aligned: compact alignment string over hypothesis
            - char_alignment: {h_line, op_line, gt_line} lists
            - mapped_reference: word-level reference segments
            - mapped_hypothesis: word-level hypothesis segments
            - mapped_ops: operation symbols grouped per reference word
    """
    dp = _edit_distance_dp(hypothesis, reference)
    ops = _backtrack_symbols(hypothesis, reference, dp)

    # Compact alignment string (matches the original SemaScore format):
    # match -> hypothesis char, substitution -> '$', insertion -> '+',
    # deletion -> '-'.
    hi = 0
    aligned_chars = []
    for op in ops:
        if op == "=":
            aligned_chars.append(hypothesis[hi])
            hi += 1
        elif op == "$":
            aligned_chars.append("$")
            hi += 1
        elif op == "-":
            aligned_chars.append("-")
            hi += 1
        elif op == "+":
            aligned_chars.append("+")
    aligned = "".join(aligned_chars)

    columns = _build_columns(reference, hypothesis, ops)

    # Build three-line character alignment for display.
    h_line = []
    op_line = []
    gt_line = []
    for h_char, op, gt_char in columns:
        h_line.append(h_char if h_char is not None else "·")
        op_line.append(op)
        gt_line.append(gt_char if gt_char is not None else "·")

    mapped_reference, mapped_hypothesis, mapped_ops = _map_to_word_segments(
        reference, hypothesis, columns
    )

    # Match-error-rate term (1 - MER) used by the original SemaScore.
    mismatches = sum(1 for op in ops if op in ("+", "-", "$"))
    mer = 1.0 - mismatches / max(len(reference), len(hypothesis), 1)

    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "aligned": aligned,
        "char_alignment": {
            "h_line": h_line,
            "op_line": op_line,
            "gt_line": gt_line,
        },
        "mapped_reference": mapped_reference,
        "mapped_hypothesis": mapped_hypothesis,
        "mapped_ops": mapped_ops,
        "mer": round(mer, 6),
    }


def format_char_alignment(char_alignment, token_width=3):
    """Pretty-print the character alignment for terminal display."""
    h_line = char_alignment["h_line"]
    op_line = char_alignment["op_line"]
    gt_line = char_alignment["gt_line"]
    fmt = lambda seq: "".join(f"{c:^{token_width}}" for c in seq)
    return f"H  : {fmt(h_line)}\nOP : {fmt(op_line)}\nGT : {fmt(gt_line)}"


if __name__ == "__main__":
    print("=== English example ===")
    result = compute_mapping(
        "I want to have a sandwich", "I vant to havea sand wich"
    )
    print("aligned:", result["aligned"])
    print(format_char_alignment(result["char_alignment"]))
    print("mapped GT:", result["mapped_reference"])
    print("mapped H :", result["mapped_hypothesis"])
    print("mapped OP:", result["mapped_ops"])
    print()

    print("=== Chinese example ===")
    result = compute_mapping("我想吃一个三明治", "我吃三民志")
    print("aligned:", result["aligned"])
    print(format_char_alignment(result["char_alignment"]))
    print("mapped GT:", result["mapped_reference"])
    print("mapped H :", result["mapped_hypothesis"])
    print("mapped OP:", result["mapped_ops"])
    print()

    print("=== Mixed Chinese-English example ===")
    result = compute_mapping(
        "在计算机系IEEE班AI班", "在计算机系ITO一班AI班"
    )
    print("aligned:", result["aligned"])
    print(format_char_alignment(result["char_alignment"]))
    print("mapped GT:", result["mapped_reference"])
    print("mapped H :", result["mapped_hypothesis"])
    print("mapped OP:", result["mapped_ops"])

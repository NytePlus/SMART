# Rubric-Based Semantic Consequence Evaluation

This directory evaluates each aligned ASR error with a configuration-driven
rubric. The research mainline is a theory-grounded eight-question Chinese
rubric; the four-question Chinese rubric remains a compact baseline. The LLM
supplies structured item answers and extractive evidence; Python applies the
final `CONSEQUENTIAL`, `INCONSEQUENTIAL`, or `UNSCORABLE` label.

The LLM is never asked to provide the formal final label.

## Visual Rubric Viewer

Open the standalone viewer directly in a browser:

```text
viewer/index.html
```

It provides the evaluation flow, theoretical mapping, searchable question
definitions, reason codes, evidence constraints, and deterministic aggregation
rules. No server or API key is required. Regenerate it after changing the JSON
rubric:

```bash
.venv/bin/python rubric-based/scripts/build_rubric_viewer.py
```

The generated HTML embeds the rubric config hash, so the displayed version can
be traced back to the exact JSON configuration.

## Theory-Grounded Chinese Rubric

`config/rubric_semantic_theory_v1.json` defines `semantic-theory-v1.0.0`:

- `G1-G2`: alignment validity and evidence sufficiency;
- `S1`: reference and lexical meaning;
- `S2`: event and proposition frame;
- `S3`: semantic roles and relations;
- `S4`: logical force, scope, and construal;
- `C1`: hypothesis-side contextual recoverability;
- `C2`: reader interpretation consequence.

The decomposition draws on lexical and cognitive semantics, frame/event
semantics, semantic-role analysis, formal semantics, and pragmatic reader
interpretation. It does not claim that the literature directly supplies the
final aggregation rule. The rationale, references, limitations, and validation
experiment are documented in `RUBRIC_RATIONALE_zh.md`.

When S1-S4 reports a semantic change, strict validation requires extractive
evidence from both the reference and hypothesis sides. C1 is prevented from
citing reference-only evidence in code.

## Recommended Minimal Rubric

`config/rubric_minimal_v1.json` defines `minimal-v1.0.0`. Its system prompt,
questions, answer options, definitions, reason codes, and boundary examples are
all written in Chinese.

- `Q0`: 当前样本是否可评分；
- `Q1`: 当前错误是否改变内容；
- `Q2`: 只看 hypothesis 和上下文能否恢复原意；
- `Q3`: 是否产生实质性读者理解偏差。

Only Q0-Q3 affect the final label. Optional Chinese `change_types` such as
`专业术语`, `数值或时间`, and `语义角色或指代` are retained for analysis but
do not vote in aggregation.

The older English eight-question rubric remains available for comparison:

```text
config/rubric_v1.json
```

## Detailed Rubric

`config/rubric_v1.json` defines detailed rubric `v1.0.2`:

- `G1`: alignment validity
- `G2`: evidence sufficiency
- `S1`: proposition or event change
- `S2`: entity, attribute, term, or value change
- `S3`: logical force or scope change
- `S4`: semantic role or relation change
- `C1`: contextual recoverability
- `C2`: reader interpretation impact

Each item has fixed answers, definitions, reason codes, allowed evidence
sources, and evidence requirements. Evidence quotes must be exact substrings of
the named input fields. `C1` cannot cite reference-only fields.

Aggregation rules are ordered and stored in the same config. The main positive
rule requires a semantic change, incomplete recoverability, and material or
critical reader impact. Invalid alignment, insufficient evidence, unresolved
semantic uncertainty, and inconsistent traces produce `UNSCORABLE`.

## Local Data

The source 300-sample file was copied and filtered locally. All 100 AISHELL-4
records were removed because of their lower quality.

```text
data/llm_classification_input_fixed300_without_aishell4.jsonl
```

The resulting 200 records contain:

- 100 `canvas-speech`
- 100 `TEDLIUM_release-3`
- 45 human consequential labels
- 155 human inconsequential labels

The `data/` and `outputs/` directories are ignored by Git.

## Validate

Run from the repository root:

```bash
.venv/bin/python rubric-based/scripts/validate_dataset.py \
  --input rubric-based/data/llm_classification_input_fixed300_without_aishell4.jsonl

.venv/bin/python -m unittest discover -s rubric-based/tests -v
```

## Human Annotation

A blind, balanced 40-record annotation template has been generated locally:

```text
data/human_annotation_minimal_v1_40.jsonl
```

It contains no previous `human_annotation` field. Follow
`HUMAN_ANNOTATION_GUIDE_zh.md`: first record an independent overall label, then
answer Q0-Q3 and choose optional change types. Validate completed annotations:

```bash
.venv/bin/python rubric-based/scripts/validate_human_annotations.py \
  --input rubric-based/data/human_annotation_minimal_v1_40.jsonl \
  --require-complete
```

For the theory-grounded rubric, use the corresponding 40-record blind template
and Chinese guide:

```text
data/human_annotation_semantic_theory_v1_40.jsonl
HUMAN_ANNOTATION_GUIDE_THEORY_zh.md
```

Validate it with:

```bash
.venv/bin/python rubric-based/scripts/validate_human_annotations.py \
  --input rubric-based/data/human_annotation_semantic_theory_v1_40.jsonl \
  --rubric rubric-based/config/rubric_semantic_theory_v1.json \
  --require-complete
```

The original holistic human labels can evaluate final-label agreement, but
item-level human annotation is required to validate the rubric itself. See
`RUBRIC_RATIONALE_zh.md` for the staged validation design.

## LLM Configuration

By default, `run_rubric.py` loads the repository `.env` file. It resolves:

```text
LLM_BASE_URL
LLM_MODEL
LLM_API_KEY or LLM_API_KEY_MASKED
```

No API key is stored in this directory or in an experiment output.

## Small Pilot

Create a deterministic subset stratified by dataset and human final label:

```bash
.venv/bin/python rubric-based/scripts/make_stratified_subset.py \
  --input rubric-based/data/llm_classification_input_fixed300_without_aishell4.jsonl \
  --output rubric-based/data/pilot20.jsonl \
  --per-stratum 5 \
  --seed 42
```

Run inference:

```bash
.venv/bin/python rubric-based/scripts/run_rubric.py \
  --input rubric-based/data/pilot20.jsonl \
  --output rubric-based/outputs/pilot20/rubric_results.jsonl \
  --workers 2 \
  --temperature 0 \
  --max-attempts 2
```

The command above uses the Chinese minimal rubric by default. To run the older
detailed rubric, add:

```text
--rubric rubric-based/config/rubric_v1.json
```

To run the theory-grounded Chinese rubric, use:

```bash
.venv/bin/python rubric-based/scripts/run_rubric.py \
  --input rubric-based/data/pilot20.jsonl \
  --output rubric-based/outputs/pilot20_theory_v1/rubric_results.jsonl \
  --rubric rubric-based/config/rubric_semantic_theory_v1.json \
  --workers 2 \
  --temperature 0 \
  --max-attempts 2
```

The theory prompt is longer than the minimal prompt. Start with 4-10 records
and inspect item traces and token usage before a larger run.

The runner appends results as they finish and resumes by `sample_id`. Failed
samples retain every raw response and validation error. Retry only failures:

```bash
.venv/bin/python rubric-based/scripts/run_rubric.py \
  --input rubric-based/data/pilot20.jsonl \
  --output rubric-based/outputs/pilot20/rubric_results.jsonl \
  --workers 1 \
  --temperature 0 \
  --max-attempts 2 \
  --retry-failures
```

## Evaluate

```bash
.venv/bin/python rubric-based/scripts/evaluate_rubric.py \
  --input rubric-based/outputs/pilot20/rubric_results.jsonl \
  --output-dir rubric-based/outputs/pilot20/report
```

Outputs:

```text
rubric_results.jsonl          # input, rubric trace, raw response, final label
rubric_results.manifest.json  # model, versions, parameters, hashes, timestamp
report/metrics.json           # final-label metrics and dataset breakdowns
report/summary.md             # compact report
report/disagreements.jsonl    # human/rubric disagreements with full traces
```

## Chinese Minimal-Rubric Smoke Test

The Chinese minimal rubric was exercised through the configured API on four
balanced records (two datasets and two human labels per class):

```text
outputs/minimal_smoke4_v1/rubric_results.jsonl
outputs/minimal_smoke4_v1/report/
```

| Metric | Value |
|---|---:|
| Inference success | 4/4 |
| Scorable coverage | 100% |
| Accuracy | 100% |
| Macro-F1 | 100% |
| Consequential recall | 100% |

This is a pipeline smoke test, not an accuracy claim: four records are far too
few for model comparison. Two records needed one automatic correction. The
validator caught an evidence source mismatch in one response, and a combined
reason code plus an evidence source mismatch in the other. Both corrected
responses then passed strict schema and extractive-evidence validation.

The successful final attempts used 21,001 tokens in total, about 5,250 tokens
per record. Including the two rejected attempts, the API returned 30,690 tokens
across six calls. Long `history` and `suffix` fields dominate prompt size, so
estimate quota from a larger pilot before starting the full dataset.

## Previous Detailed-Rubric Holdout Result

Detailed rubric `v1.0.2` was evaluated on 20 stratified records that were not used for
the preceding `v1.0.1` calibration pilot:

| Metric | Value |
|---|---:|
| Inference success | 20/20 |
| Scorable coverage | 100% |
| Accuracy | 85.0% |
| Macro-F1 | 84.96% |
| Consequential precision | 88.89% |
| Consequential recall | 80.0% |
| Cohen's kappa | 0.70 |

The three disagreements are available under
`outputs/holdout20_v1_0_2/report/disagreements.jsonl`.

This is preliminary end-to-end agreement. The source data has holistic human
labels but no human answers for G1-C2, so it cannot yet separate rubric-design
errors from LLM item-execution errors.

## Full 200-Sample Run

The current four-record minimal-rubric smoke test used approximately 5,250
tokens per successful final call, before retry overhead. This would be roughly
1.05 million tokens for 200 first-pass calls if the sample remains
representative; retries increase that total. Run and inspect a 20-record pilot
before launching all 200 records.

```bash
.venv/bin/python rubric-based/scripts/run_rubric.py \
  --input rubric-based/data/llm_classification_input_fixed300_without_aishell4.jsonl \
  --output rubric-based/outputs/full200_minimal_v1/rubric_results.jsonl \
  --workers 2 \
  --temperature 0 \
  --max-attempts 2

.venv/bin/python rubric-based/scripts/evaluate_rubric.py \
  --input rubric-based/outputs/full200_minimal_v1/rubric_results.jsonl \
  --output-dir rubric-based/outputs/full200_minimal_v1/report
```

Do not revise the rubric against the same records later reported as a frozen
test set. The current 200 records should be treated as development data unless
a separate frozen test split is defined.

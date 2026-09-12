# ASR Post-Correction

一个不在推理阶段读取 GT 的、rubric 引导的单轮/多轮 ASR 后修正框架。每轮严格执行：

1. 无损拆分 hypothesis；
2. 仿照 `semantic-error-rate/rubric-based` 的 G/S/C 分解定位可疑 segment；
3. 使用当前 hypothesis 与同一 session/talk 的上文产生保守修正建议；
4. 通过确定性 offset、唯一性、证据和置信度门控应用编辑。

多轮模式在无候选、无实际编辑、已见状态、`A→B→A` 二循环或达到轮数上限时停止。二循环的反向编辑不会被接受。

## 安装与测试

优先使用项目已有环境；远端当前可用 `semantic-error-rate/.venv`：

```bash
cd /hpc_stor03/sjtu_home/xiaoyu.gu/workspace/wcc_workspace/asr_post_correction
/hpc_stor03/sjtu_home/xiaoyu.gu/workspace/semantic-error-rate/.venv/bin/pip install -e .
/hpc_stor03/sjtu_home/xiaoyu.gu/workspace/semantic-error-rate/.venv/bin/python -m unittest discover -s tests -v
```

密钥只从 `LLM_API_KEY` 或 `OPENAI_API_KEY` 环境变量读取，不写入仓库、manifest、cache 或结果。

## 固定 prompt pilot

### GT-context 单步 baseline

`gt-context-baseline` 是单独的 GT-history 实验入口，不启用 verifier、热词、alternate 或 deterministic postprocessor。旧式 prompt 将 benchmark 保存的 `history` 与当前 hypothesis 传给一次 LLM；普通 `prompts/v1.json` 则先运行无当前句 reference 的 `S1-S4/C1-C2` 候选检测，再把通过精确 span 校验的候选、同一 history 和 hypothesis 传给修正器。两阶段都只允许返回当前句子的局部结果，当前句 target 不进入任何 LLM payload。由于 history 包含 ground truth，这个入口仍必须视为 oracle/泄漏实验，manifest 会记录 `ground_truth_context_visible=true`，结果不能解释为可部署的无 GT 系统。

```bash
$PY -m asr_post_correction.cli gt-context-baseline \
  --data-prepare-root $DP \
  --model-key qwen3-asr \
  --output outputs/gt_context_baseline_v1_qwen3-asr.jsonl \
  --prompt prompts/gt_context_baseline_v1.json \
  --workers 4
```

对 SenseVoice 将 `--model-key` 改为 `sensevoice`。运行前通过 `OPENAI_API_KEY` 或 `LLM_API_KEY` 注入密钥；密钥不得写入文件。

### 权威 WER / c-SEER2

```bash
$PY -m asr_post_correction.cli headline-evaluate \
  --data-prepare-root $DP --model-key qwen3-asr \
  --corrected outputs/gt_context_baseline_v1_qwen3-asr.jsonl \
  --output-dir outputs/gt_context_baseline_v1_qwen3-asr_wer

$PY -m asr_post_correction.cli prepare-headline-cseer \
  --data-prepare-root $DP --model-key qwen3-asr \
  --corrected outputs/gt_context_baseline_v1_qwen3-asr.jsonl \
  --model-dir outputs/gt_context_baseline_v1_qwen3-asr_cseer
# 然后运行 data-prepare 的 run_llm_classification.py，完成后：
$PY -m asr_post_correction.cli finalize-headline-cseer \
  --data-prepare-root $DP \
  --model-dir outputs/gt_context_baseline_v1_qwen3-asr_cseer \
  --output outputs/gt_context_baseline_v1_qwen3-asr_cseer.json
```

`headline-evaluate` 只调用 `data-prepare/benchmark/testing/scripts/asr_testing/wer.py`；`finalize-headline-cseer` 只调用 `data-prepare/scripts/26.8.2/_ser_metrics.py`，输出权威 `c_ser2`。

成对比较使用 `scripts/paired_metric_statistics.py`。除 `old` 和 `after` 外，
它还输出同分母的 `repaired` 与 `new` 率及各自的 paired-utterance bootstrap
95% CI，并逐条校验 `after = old - repaired + new`。WER 错误按 reference
token 位置或插入边界匹配；c-SEER2 semantic hypothesis-error chunks 按
reference 局部 span 匹配。显著性字段仍是 Old 与 After 的双侧 paired
randomization test。

每个输出 manifest 记录 prompt version 与 SHA-256。修改 prompt 必须新建版本文件，不覆盖已经实验过的版本。

```bash
export LLM_API_KEY='...'
PY=/hpc_stor03/sjtu_home/xiaoyu.gu/workspace/semantic-error-rate/.venv/bin/python
ROOT=/hpc_stor03/sjtu_home/xiaoyu.gu/workspace

$PY -m asr_post_correction.cli run \
  --input $ROOT/data-prepare/model-test/canvas-speech/qwen3-asr-1.7b/pred.raw.jsonl \
  --message-file $ROOT/data-prepare/data-post-processing/canvas-speech/processed/message.jsonl \
  --prompt prompts/v1.json --mode single --limit 40 --workers 2 \
  --output outputs/v1/canvas_qwen3_single.jsonl

$PY -m asr_post_correction.cli evaluate \
  --data-prepare-root $ROOT/data-prepare \
  --before $ROOT/data-prepare/model-test/canvas-speech/qwen3-asr-1.7b/pred.raw.jsonl \
  --corrected outputs/v1/canvas_qwen3_single.jsonl \
  --output-dir outputs/v1/canvas_qwen3_single_analysis \
  --require-complete
```

若 `--input` 是抽样子集，必须同时传 `--context-input` 指向完整原始预测，避免把中间未抽中的上文错误地丢掉。该文件仍只提供 hypothesis，不向推理引擎暴露 GT。

分析目录把所有实际变化分成 `improved.jsonl`、`regressed.jsonl`、`neutral.jsonl`，每条保留完整 rubric 与修正 trace。下一版 prompt 的变更说明必须引用这些错例的聚合模式，不能根据直觉修改。

## 纯 LLM 内容纠错

要排除 normalize、标点、空格、大小写和撇号规范化带来的 WER 收益，运行时必须使用 `--llm-only`。该模式要求 prompt 的 `deterministic_postprocessors` 为空，并在应用层拒绝内容签名不变的格式编辑。v8 GT-context baseline 不启用 OCR、alternate 或其他外部证据。

v8 GT-context baseline 只使用句前上下文与当前 hypothesis，不启用 OCR、alternate
或其他外部证据。

`evaluate` 没有项目内 WER 实现；它强制加载 `data-prepare/benchmark/testing/scripts/asr_testing/wer.py`。需要同时启用完整性和纯 LLM 审计时：

```bash
$PY -m asr_post_correction.cli evaluate \
  --data-prepare-root $ROOT/data-prepare \
  --before inputs/dev_semantic20_canvas-speech_qwen3.jsonl \
  --corrected outputs/v7/dev_semantic20_canvas_qwen3.jsonl \
  --output-dir outputs/v7/dev_semantic20_canvas_qwen3_official_analysis \
 --require-complete --require-llm-only
```

该命令会拒绝缺失行、API 失败行、预先改写过的 `original_hypothesis`、含 postprocessor 的 manifest，以及任何确定性或仅格式变化的 applied edit。

alternate 相关命令仅保留为历史实验接口，不属于 v8 baseline，不能与 v8 结果混用。

## 全量与 c-SEER

目标表中的 headline WER 使用 8 个 X/Y/Z benchmark bucket，排除 AISHELL-4，
不按 source 去重，保留 Canvas/TED composite 的重复权重，并对 467 条记录统一
做 micro-average。以下命令直接复用 `data-prepare` 的 T1 筛选链路和 WER：

```bash
$PY -m asr_post_correction.cli headline-evaluate \
  --data-prepare-root $ROOT/data-prepare \
  --model-key qwen3-asr \
  --output-dir outputs/headline/qwen3_baseline

$PY -m asr_post_correction.cli headline-evaluate \
  --data-prepare-root $ROOT/data-prepare \
  --model-key sensevoice \
  --output-dir outputs/headline/sensevoice_baseline
```

命令固定校验 467 条和 18,284 个 reference token；当前权威基线分别为
Qwen3-ASR `12.9730912273%`、SenseVoice `11.8683001531%`。传入
`--corrected ... --require-llm-only` 后，会在同一 scope 上评估修正结果，并要求
每条 `original_hypothesis` 与 `data-prepare` 原预测在官方 WER token 序列上完全
一致。报告同时保留原始字符串差异计数，以显式记录 SenseVoice 控制标签被输入层
移除的情况。

先将结果导出为现有评价体系的输入：

```bash
$PY -m asr_post_correction.cli export \
  --corrected outputs/v1/canvas_qwen3_multi.jsonl \
  --output outputs/v1/canvas_qwen3_multi_eval/pred.raw.jsonl
```

`export` 只写入成功行的 `id,target,pred`，并将转换输入中的
`canvas-speech:0000001` 统一导出为数据集本地 ID `0000001`，以便
data-prepare 的上下文和 c-SEER 脚本正确关联记录。

然后在 `data-prepare` 中对修正前、修正后分别运行同一组 `build_segment_mapping.py`、`build_llm_classification_input.py`、`run_llm_classification.py`，保证 c-SEER 口径完全一致。最后复用权威实现：

```bash
$PY -m asr_post_correction.cli cseer \
  --data-prepare-root $ROOT/data-prepare \
  --model-dir outputs/v1/canvas_qwen3_multi_eval
```

`model-dir` 必须包含 `pred.raw.jsonl`、`llm_classification_input.jsonl`、`results/classification_results.jsonl`。c-SEER 的值是输出中的 `c_ser`。禁止用自定义近似指标替代目标表中的 6.40/6.23 口径。

## 实验纪律

- correction engine 收不到 `target`/`ground_truth`；GT 仅由离线指标和错例分析读取。
- prompt 不包含数据集专名词表、样本 ID、测试句模式或基于测试集答案构造的规则。
- 每一版先冻结 prompt/hash，再测试，再按 improved/regressed 证据提出下一版。
- 全量结果必须报告排除 AISHELL 后的 467 条 headline micro-WER，并另外分别报告 Canvas/TED、Qwen3-ASR/SenseVoice、单轮/多轮的覆盖率、失败数、WER、c-SEER 与相对降幅。

# Benchmark metrics scripts (2026-08-02)

SER/WER 聚合与表格/样例生成脚本（含 source 去重）。

## 脚本

| 文件 | 用途 |
|---|---|
| `aggregate_benchmark_full_metrics.py` | 全模型 SER / X·Z 均值，输出 `full_metrics.json` + LaTeX |
| `evaluate_wer.py` | 计算 benchmark WER，输出 `wer_metrics.json` + LaTeX |
| `generate_metrics_table_t1.py` | 生成表一：全量 benchmark WER + s-SER / c-SER / w-SER（source 去重） |
| `generate_metrics_tables_t3_t4.py` | 生成表三、表四及 E6（排除 AISHELL，source 去重） |
| `generate_metrics_table_t5.py` | 生成表五：全量 benchmark 每个 jieba chunk 含 word 数分布（中文按字、英文按词） |
| `generate_metrics_table_t2_segment_chunks.py` | T2 变体：三组 WER 模型每个 error segment 含 chunk 数 |
| `generate_metrics_examples.py` | 生成样例 E1–E6 |
| `analyze_wer_sser_divergence.py` | 校验分类结果一致性，分析 WER 高但 s-SER 低的模型反转与 case |

依赖 `benchmark/testing/scripts/asr_testing/` 包（通过 `_path_setup.py` 自动加入 `PYTHONPATH`）。

## 示例

```bash
cd /path/to/data-prepare
PY=~/miniconda3/envs/detailed-semantic/bin/python

$PY scripts/26.8.2/evaluate_wer.py
$PY scripts/26.8.2/aggregate_benchmark_full_metrics.py --skip-xz --partial
$PY scripts/26.8.2/generate_metrics_table_t1.py
$PY scripts/26.8.2/generate_metrics_examples.py
$PY scripts/26.8.2/generate_metrics_tables_t3_t4.py
$PY scripts/26.8.2/generate_metrics_table_t5.py
$PY scripts/26.8.2/generate_metrics_table_t2_segment_chunks.py
$PY scripts/26.8.2/analyze_wer_sser_divergence.py
```

`_ser_metrics.py` 会校验当前 `llm_classification_input.jsonl` 与分类结果的
`target/pred/reference_segment/hypothesis_segment/op_segment` 指纹；发现旧结果复用同一
`(id, focus_id)` 时会停止聚合，避免静默生成错误的 SER。

输出默认写入 `benchmark/testing/reports/asr_full/`，结构见该目录下 `README.md`。

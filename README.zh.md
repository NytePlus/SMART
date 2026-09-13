# Smart Pipeline

Smart Pipeline 提供四个 Python 入口，分别用于 ASR segment mapping、基于 rubric
的语义后果评估、SIER 指标聚合，以及只使用历史上下文的 ASR 后修正。

[English documentation / 英文文档](README.md)

## 并列的两条流水线

后修正流水线和评估流水线是相互独立、在仓库根目录并列的两个组件。后修正推理
不会接收当前句的 reference；reference 只进入离线评估流水线。

```text
smart-pipeline/
├── post_correction/                  # 后修正实现和 prompt
├── segment_mapping/                  # 评估：reference/hypothesis 对齐
├── rubric-based/                     # 评估：唯一 rubric 配置和推理
├── sier_metrics/                     # 评估：SIER 指标聚合
├── run_post_correction.py            # 流水线 1：只使用历史的后修正
├── run_segment_mapping.py            # 流水线 2a：segment mapping
├── run_rubric_sier.py                # 流水线 2b：rubric 推理和 SIER
├── run_evaluation.py                 # 流水线 2：完整评估
├── requirements.txt                  # 运行依赖
├── Dockerfile                         # 可复现的容器镜像配置
├── .dockerignore                      # Docker 构建上下文排除项
├── docker/jieba-cache/.gitkeep        # 可写缓存目录占位
├── README.md
└── README.zh.md
```

```mermaid
flowchart LR
    subgraph Evaluation[评估流水线]
        E1["reference + hypothesis"] --> M["Segment Mapping"]
        M --> E2["对齐后的 (r_k, h_k) + c_k"]
        E2 --> R["Reference-aware evaluation rubric<br/>Reference-aware contextual semantic consequence judgment"]
        R --> G["结构化判断 (G_k, S_k, C_k)"]
        G --> A["确定性聚合"]
        A --> E3["consequential / inconsequential"]
    end

    subgraph Correction[后修正流水线]
        C1["(h, c)"] --> D["Detector<br/>Target-reference-free rubric"]
        D --> DR["repair-oriented guidance 𝒟^rep"]
        DR --> C["Corrector LLM<br/>Target-reference-free rubric-guided correction"]
        C --> C2["修正后的 hypothesis h̃"]
    end

    Evaluation ~~~ Correction

    classDef plainText fill:transparent,stroke:transparent,stroke-width:0px
    class E1,E2,G,E3,C1,DR,C2 plainText

    style Evaluation fill:transparent,stroke:#7f8c8d,stroke-width:1px,stroke-dasharray:6 4,rx:10px,ry:10px
    style Correction fill:transparent,stroke:#7f8c8d,stroke-width:1px,stroke-dasharray:6 4,rx:10px,ry:10px
```

## 安装

使用 Python 3.10 或更高版本，并安装共享依赖：

```bash
python -m pip install -r requirements.txt
```

Rubric 和后修正调用 OpenAI 兼容 API。通过环境变量设置 `LLM_API_KEY`（或
`OPENAI_API_KEY`）；密钥不会从 JSONL 输入读取，也不会写入结果文件。

## Docker 环境

仓库保留了基于 Python 3.13 的 Docker 配置。构建镜像并运行不依赖 API 的入口检查：

```bash
docker build -t smart-pipeline .
docker run --rm smart-pipeline run_segment_mapping.py --help
```

如果需要使用离线或本地缓存的 Python 基础镜像，可以覆盖
`PYTHON_BASE_IMAGE`：

```bash
docker build \
  --build-arg PYTHON_BASE_IMAGE=python:3.11-slim-bookworm \
  -t smart-pipeline .
```

运行流水线时挂载输入和输出目录。需要 API 的命令还要通过 `--env` 传入
`LLM_API_KEY` 或 `OPENAI_API_KEY`。

## 四个入口

### 1. Segment mapping

输入是 UTF-8 JSONL，每行一个对象。稳定输入 schema 为：

```json
{
  "id": "sample-001",
  "reference": "参考转写文本",
  "hypothesis": "ASR 识别文本"
}
```

运行确定性的对齐阶段：

```bash
python run_segment_mapping.py \
  --input path/to/evaluation_input.jsonl \
  --output path/to/segment_mapping.jsonl
```

稳定输出 schema 为：

```json
{
  "id": "sample-001",
  "reference": "参考转写文本",
  "hypothesis": "ASR 识别文本",
  "mapping": {
    "mapped_reference": ["参考", "转写", "文本"],
    "mapped_hypothesis": ["参考", "撰写", "文本"],
    "mapped_ops": ["==", "SS", "=="],
    "mer": 0.333333,
    "aligned": "..."
  },
  "error_segments": [
    {
      "segment_index": 1,
      "reference_segment": "转写",
      "hypothesis_segment": "撰写",
      "op_segment": "SS",
      "modified_reference": "参考撰写文本"
    }
  ]
}
```

`mapped_*` 数组描述完整对齐；`error_segments` 只包含发生替换、插入或删除的
segment。`modified_reference` 是将选定 hypothesis segment 应用到 reference 后的文本。

### 2. Rubric 评估与 SIER

仓库只分发一个 rubric：`rubric-based/config/rubric_v1.json`。它定义问题、可选答案、
证据规则，以及 `CONSEQUENTIAL`、`INCONSEQUENTIAL` 和 `UNSCORABLE` 的确定性聚合规则。

```bash
python run_rubric_sier.py \
  --input path/to/segment_mapping.jsonl \
  --rubric rubric-based/config/rubric_v1.json \
  --segment-output path/to/rubric_results.jsonl \
  --metric-output path/to/sier_metrics.json \
  --workers 4
```

命令只把对齐后的错误 segment 发送给 judge，写出逐 segment 的标准化标签，并计算权威
的 `s_ser`、`c_ser`、`c_ser2`、`w_ser` 和 `w_mer2` 变体。失败或无效的推理会记录为
`UNSCORABLE`。

逐 segment 输出 schema 为：

```json
{
  "id": "sample-001",
  "segment_index": 1,
  "reference_segment": "转写",
  "hypothesis_segment": "撰写",
  "op_segment": "SS",
  "label": "CONSEQUENTIAL",
  "is_semantic_error": true,
  "rubric_version": "v1.0.2",
  "items": {
    "G1": {
      "answer": "有效",
      "reason_code": "...",
      "confidence": 0.98,
      "evidence": []
    }
  },
  "change_types": []
}
```

`change_types` 仅用于分析，不参与最终 label 的聚合。

SIER 汇总输出 schema 为：

```json
{
  "utterances": 100,
  "total_target_chunks": 1200,
  "total_target_words": 1850,
  "error_spans": 140,
  "semantic_error_spans": 52,
  "sier": 4.333333,
  "variants": {
    "s_ser": 4.333333,
    "c_ser": 3.916667,
    "c_ser2": 4.083333,
    "w_ser": 3.135135,
    "w_mer2": 3.405405
  },
  "unscorable_segments": 2
}
```

`sier` 是作为主指标暴露的 `s_ser` 值；所有权威变体都会保留在 `variants` 中。

### 3. 后修正

后修正输入不能包含 `reference` 或 `target`：

```json
{
  "id": "sample-001",
  "history_context": "同一会话中当前句之前的文本",
  "hypothesis": "当前 ASR 输出"
}
```

使用唯一的 canonical prompt 运行只依赖历史的后修正引擎：

```bash
python run_post_correction.py \
  --input path/to/correction_input.jsonl \
  --output path/to/corrected.jsonl \
  --prompt post_correction/prompts/v1_1.json \
  --mode single \
  --workers 4
```

使用 `--mode multi` 可启用有上限的多轮修正。每条输出记录原始和修正后的 hypothesis、
停止原因、各轮 trace 以及见过的状态，便于审计。

输出 schema 为：

```json
{
  "id": "sample-001",
  "status": "ok",
  "original_hypothesis": "当前句的 ASR 识别文本",
  "corrected_hypothesis": "修正后的 ASR 文本",
  "stop_reason": "no_candidates",
  "rounds": [],
  "seen_hypotheses": ["当前句的 ASR 识别文本"]
}
```

`rounds` 保留每轮的 segments、候选错误、修正建议、验证结果、实际应用编辑与拒绝编辑。

### 4. 完整评估

`run_evaluation.py` 依次调用 segment mapping、rubric 推理和 SIER 聚合，不调用后修正：

```bash
python run_evaluation.py \
  --input path/to/evaluation_input.jsonl \
  --output-dir path/to/evaluation_output \
  --rubric rubric-based/config/rubric_v1.json \
  --workers 4
```

输出目录包含：

```text
segment_mapping.jsonl
rubric_results.jsonl
sier_metrics.json
manifest.json
```

`manifest.json` 记录解析后的输入和 rubric 路径、模型参数以及三个生成文件的路径：

```json
{
  "created_at": "2026-01-01T00:00:00+00:00",
  "input": "/path/to/evaluation_input.jsonl",
  "rubric": "/path/to/rubric-based/config/rubric_v1.json",
  "model": "deepseek-v4-pro",
  "outputs": {
    "segment_mapping": "/path/to/evaluation_output/segment_mapping.jsonl",
    "rubric_results": "/path/to/evaluation_output/rubric_results.jsonl",
    "sier_metrics": "/path/to/evaluation_output/sier_metrics.json"
  }
}
```

如需比较修正前后的效果，先运行后修正，再用其 `corrected_hypothesis` 构造新的评估
JSONL，分别对原始和修正后的 hypothesis 运行完整评估。

## 数据约定

- 批量输入和中间结果使用 UTF-8 JSONL。
- 数据集级指标和 manifest 使用 JSON。
- 对外字段为 `reference` 和 `hypothesis`；指标适配器只在临时工作目录内部使用
  `target` 和 `pred`。
- 每个输入文件中的 `id` 必须唯一。
- 运行时按需创建输出和缓存目录，实验结果应放在源码目录之外。

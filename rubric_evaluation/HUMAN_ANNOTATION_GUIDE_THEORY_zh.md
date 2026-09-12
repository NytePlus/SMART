# 理论驱动 Rubric 人工标注说明

标注文件：

```text
data/human_annotation_semantic_theory_v1_40.jsonl
```

模板不包含原有 `human_annotation` 二分类标签。每行只评价当前
`reference_segment` 与 `hypothesis_segment` 对应的一个错误。

## 标注顺序

先填写 `preliminary_overall_label` 和 `preliminary_overall_confidence`，再回答 rubric。
整体标签填写后不要根据后续题目回改，它用于独立检验聚合规则。

然后依次回答：

- `G1`：span 是否对应同一局部编辑位置；
- `G2`：证据是否足以评分；
- `S1`：指称对象、词义、属性、术语或值是否改变；
- `S2`：事件、状态、谓词或命题是否改变；
- `S3`：参与者角色、指代或语义关系是否改变；
- `S4`：否定、模态、范围、时体、焦点或视角是否改变；
- `C1`：只看 hypothesis 和上下文能否恢复原意；
- `C2`：读者最终理解偏差的程度。

S1-S4 可以同时为“语义改变”。不要为了让变化类型唯一而压制真实重叠。

## Evidence

每条证据格式为：

```json
{"source": "reference_span", "quote": "逐字复制的原文"}
```

S1-S4 判为“语义改变”时，必须至少引用一条 reference 侧证据和一条 hypothesis 侧
证据。C1 只能引用 `hypothesis_span`、`hypothesis_text`、`history` 或 `suffix`，不能
引用 reference。所有 quote 必须是对应字段的精确子串。纯插入或纯删除导致
`reference_span` 或 `hypothesis_span` 本身为空时，可以对这个空 span 填写空 quote
表示缺位；非空字段和其他来源不得使用空 quote。

## 完成与校验

填写 `annotator_id`，将 `status` 改为 `已完成`，然后运行：

```bash
.venv/bin/python rubric-based/scripts/validate_human_annotations.py \
  --input rubric-based/data/human_annotation_semantic_theory_v1_40.jsonl \
  --rubric rubric-based/config/rubric_semantic_theory_v1.json \
  --require-complete
```

具体选项、定义、原因代码和边界示例以
`config/rubric_semantic_theory_v1.json` 为准。方法依据和实验设计见
`RUBRIC_RATIONALE_zh.md`。

# Minimal Rubric 人工标注说明

标注文件：

```text
data/human_annotation_minimal_v1_40.jsonl
```

文件中的旧人工二分类标签已经移除。请不要从其他文件查询原标签。

每行是一个错误 segment。主要阅读字段：

- `target`：完整 reference；
- `pred`：完整 ASR hypothesis；
- `reference_segment`：当前 reference 错误片段；
- `hypothesis_segment`：对应 hypothesis 片段；
- `structured_prediction`：当前错误在句中的位置；
- `history`、`suffix`：允许使用的上下文。

只评价当前 `<focus>` 对应的错误，不要把句子中的其他 ASR 错误归到当前 segment。

## 标注顺序

### 第一步：独立整体判断

在阅读样本后、查看 Q0-Q3 选项前，先填写：

```json
"preliminary_overall_label": "有实质语义后果 | 无实质语义后果 | 不可评分",
"preliminary_overall_confidence": "高 | 中 | 低"
```

填写后不要因为后面的 rubric 答案而修改该字段。它用于检验 rubric
聚合结果是否符合人工直觉。

### 第二步：回答 Q0-Q3

#### Q0 可评分性

- `可评分`：局部对齐合理，reference 意图基本明确，信息足够；
- `不可评分`：对齐错误、reference 有问题、必要上下文缺失或存在无法仲裁的多种解释。

语义差异很大不代表对齐无效。例如“循环→寻”仍是可评分的 substitution。

#### Q1 内容变化

- `无内容变化`：无意义填充、重复等，不改变内容；
- `仅形式变化`：拼写、空格、词形不同，但仍表达同一内容；
- `内容变化`：事实、实体、术语、数值、逻辑、角色、关系或有效信息发生变化；
- `不确定`：无法可靠判断。

#### Q2 上下文可恢复性

- `完全可恢复`：只看 hypothesis 和上下文，普通读者仍能唯一恢复原意；
- `部分可恢复`：只能恢复主题或部分信息，关键内容仍不确定；
- `不可恢复`：会形成错误理解或存在多个实质不同解释；
- `不确定`：上下文不足以判断。

回答 Q2 时必须采用反事实视角：假设读者看不到 reference。不得因为标注者
已经知道 reference，就认为读者也能脑补缺失内容。

#### Q3 读者理解影响

- `无影响`：最终理解相同；
- `轻微影响`：局部措辞有偏差，但核心事实、身份、术语、数值、逻辑和关系不变；
- `实质影响`：至少一个上述关键信息可能被误解或无法确定；
- `不确定`：无法可靠判断影响程度。

“还能看出大致主题”不能作为轻微影响的充分条件。

### 第三步：变化类型

`change_types` 可选择零个或多个：

```text
命题或事件
实体或专名
专业术语
数值或时间
否定或模态
条件或范围
语义角色或指代
因果或时序
信息增删
其他
```

变化类型只用于分析，不决定最终标签。Q1 为“无内容变化”或“仅形式变化”时，
通常填写空列表。

## Evidence 填写

每条 evidence 使用：

```json
{"source": "reference_span", "quote": "逐字复制的原文"}
```

允许的 `source` 会在 rubric 配置中规定。`quote` 必须是对应字段的精确子串，
保留空格、大小写和标点，不要改写。

Q2 不允许引用 `reference_text` 或 `reference_span`，只能引用：

```text
hypothesis_span
hypothesis_text
history
suffix
```

## 完成状态

标注完成后填写：

```json
"status": "已完成"
```

如有边界问题，在 `final_comment` 中说明。不要删除原始输入字段，也不要改变
`sample_id`。

## 校验标注文件

完成后运行：

```bash
.venv/bin/python rubric-based/scripts/validate_human_annotations.py \
  --input rubric-based/data/human_annotation_minimal_v1_40.jsonl \
  --require-complete
```

脚本会严格检查选项、原因代码、confidence 和 evidence，并比较人工独立整体
标签与代码聚合标签。它不会调用 LLM。

# 理论驱动 Rubric 的设计依据与验证方案

## 1. 方法主张

`semantic-theory-v1.0.0` 的目标不是让 LLM 阅读更多说明后直接二分类，而是把
“ASR 错误是否造成实质理解偏差”操作化为三个可分离阶段：

```text
输入是否可评价
  -> 哪一类语义结构发生变化
  -> hypothesis 侧上下文能否恢复原意
  -> 读者理解偏差的程度
  -> 代码确定性聚合
```

形式化表示为：

```text
z = (G1, G2, S1, S2, S3, S4, C1, C2)
y = g(z)
```

LLM 只产生结构化变量 `z` 和抽取式证据，正式标签 `y` 由版本化规则 `g`
计算。理论文献为语义维度的划分提供依据，但不会自动证明问题定义、阈值或聚合
规则正确；这些内容必须通过人工标注和冻结测试集验证。

## 2. 为什么是这八个问题

### G1-G2：测量前提

局部 span 对齐错误、reference 本身错误或必要上下文缺失时，后续答案测到的不是
ASR 语义后果。G1 将“位置是否对应”与“含义是否相似”分开，G2 将证据不足显式
转为 `UNSCORABLE`，避免模型被迫猜测。

### S1：指称与词义

这个问题检查表达式指向的对象、概念、类别、属性和值。它覆盖专名、术语、数字、
日期、单位和版本，但不采用“字符串不同即语义不同”的规则。同一实体的可识别拼写
变体属于形式差异。

理论来源主要是词汇语义学、指称研究，以及认知语义学把意义视为概念化而非单纯
词形映射的观点。

### S2：事件与命题框架

这个问题检查“发生了什么/什么为真”：谓词唤起的事件或状态框架、事实和命题内容
是否被新增、删除或替换。框架语义学解释了为什么一个词的意义需要结合其唤起的
概念框架和参与结构理解；事件语义和命题语义提供了动作、状态及真值内容的分析
视角。

### S3：语义角色与关系

这个问题检查“谁对谁做了什么”：施事、受事、体验者、指代、因果、时序和观点
归属。事件类型未变并不保证关系未变，例如主客体交换仍可使用同一谓词，但会产生
不同命题。

它主要对应框架元素、语义角色和论元结构研究。

### S4：逻辑力、范围与识解

这个问题检查命题“以何种方式成立和被组织”：否定、模态、确定性、条件、比较、
量化与作用域、时态与体、焦点和观察视角。

其中否定、模态、量化和作用域来自形式语义的经典分析；焦点、视角和凸显方式借鉴
认知语义学的 `construal`（识解）概念。这里的“识解”不是泛指所有认知差异，只
保留能够由当前文本证据确认、并可能改变读者解释的部分。

### C1：上下文可恢复性

C1 是一个反事实读者测试：假设读者看不到 reference，只能阅读 ASR hypothesis、
history 和 suffix，他能否唯一恢复原意？它与“上下文是否相关”不同，也不判断信息
是否重要。

因此 C1 的证据来源在代码中被限制为 hypothesis 侧字段。若模型引用 reference，
输出会校验失败并自动重试。

### C2：读者理解后果

C2 将最终理解偏差记录为有序等级：无、轻微、实质、关键。它不是正式 final label，
因为正例仍需同时满足：样本可评分、至少一个语义结构发生变化、原意未被完全恢复，
并且理解偏差达到实质程度。

C2 与目标标签较接近，所以实验中必须增加 `C2-only` 对照。如果完整 rubric 没有比
只问 C2 更准确、更稳定或更容易定位错误，就不能声称结构化分解带来了方法增益。

## 3. 维度边界

S1-S4 允许多选式重叠，不强制把一个错误塞进唯一类别。例如：

| 错误 | 可能变化的维度 | 原因 |
|---|---|---|
| `did not improve -> improved` | S2、S4 | 结果状态和否定极性同时改变 |
| `15% -> 50%` | S1 | 具体值改变 |
| `A criticized B -> B criticized A` | S3 | 参与者角色交换 |
| 专名有轻微拼写差异且上下文唯一确定 | S1 仅形式差异、C1 完全可恢复 | 指称对象保持 |
| `may succeed -> succeeded` | S2、S4 | 状态与模态/完成性改变 |

这种重叠是分析结果，不是标签冲突。聚合器只检查 S1-S4 中是否至少一个为“语义
改变”，同时保留具体触发维度用于错误分析。

## 4. 确定性聚合

主正例规则为：

```text
G1 in {有效, 部分有效}
and G2 == 可评分
and any(S1:S4 == 语义改变)
and C1 in {部分可恢复, 不可恢复}
and C2 in {实质影响, 关键影响}
```

以下情况优先输出 `UNSCORABLE`：无效/不可判断的对齐、证据不足、C1/C2 不可
评分、没有已确认语义变化但存在未解决的语义不确定，以及内部矛盾的 trace。

所有语义维度保持不变或仅为形式差异、C2 为无/轻微影响，或语义变化被上下文完全
恢复且没有实质理解偏差时，输出 `INCONSEQUENTIAL`。

## 5. 如何验证准确性

现有 200 条数据中的人工二分类结果只能检验最终标签一致性，不能检验 S1-S4、C1、
C2 是否被正确执行。建议按以下顺序验证。

### 阶段 A：人工 construction/calibration

1. 从 development 数据中按数据集、人工标签和错误类型分层抽取 40-60 条。
2. 至少两名标注者先独立填写整体标签，再逐题填写 G1-C2 和原文证据。
3. 计算每题 Cohen's kappa；有序的 C1/C2 同时计算 weighted kappa。
4. 对低一致性题进行仲裁，记录是定义问题、证据不足还是标注执行问题。
5. 只在 construction/development 数据上修改 rubric，并提升版本号。

先填整体标签可以避免人工整体判断被聚合规则污染，也可以比较“人工直觉标签”和
“人工 rubric 聚合标签”，直接检验规则 `g` 是否合理。

### 阶段 B：LLM 执行验证

在已仲裁的逐题人工标注上报告：

- 每个 item 的 accuracy、kappa、confusion matrix；
- evidence 精确匹配率和 evidence source 合规率；
- 人工 rubric 聚合标签与 LLM rubric 聚合标签的一致性；
- 错误来自 item 执行还是聚合规则；
- `UNSCORABLE` coverage 和原因分布。

### 阶段 C：方法对照

至少比较以下配对条件，并对同一批样本使用相同 judge model 和推理参数：

| 条件 | 目的 |
|---|---|
| Direct prompt | 原项目基线 |
| C2-only | 检验完整 rubric 是否超过“换一种方式直接问标签” |
| Minimal Q0-Q3 | 检验细化语义机制是否有增益 |
| Theory S1-S4 + C1 + C2 | 主方法 |
| Theory without evidence | 检验抽取式证据约束 |
| Theory without context | 检验上下文恢复步骤 |

最终标签使用 macro-F1、consequential recall、Cohen's kappa 和 scorable coverage；
配对错误差异使用 McNemar 检验，指标置信区间使用按样本 bootstrap。另用不同 seed
重复采样和不同 judge model 检验稳定性。

### 阶段 D：冻结测试

rubric 问题、定义、例子、聚合规则、prompt、judge model 和参数全部冻结后，才运行
独立 test set。来自同一长音频、speaker、meeting 或相邻上下文的样本不能跨
development/test，以防上下文泄漏。

当前去除 AISHELL-4 的 200 条数据已经参与开发和 smoke test，应视为 development，
不能再作为最终冻结测试集来支撑论文主结论。

## 6. 主要文献脉络

以下文献用于说明分析维度的来源，不应被表述为它们直接提出了本项目的八题 rubric：

- Talmy, L. (2000). *Toward a Cognitive Semantics*. MIT Press.
- Langacker, R. W. (1987). *Foundations of Cognitive Grammar, Vol. 1: Theoretical Prerequisites*. Stanford University Press.
- Fillmore, C. J. (1968). “The Case for Case.” In *Universals in Linguistic Theory*.
- Fillmore, C. J. (1982). “Frame Semantics.” In *Linguistics in the Morning Calm*.
- Baker, C. F., Fillmore, C. J., & Lowe, J. B. (1998). “The Berkeley FrameNet Project.” *COLING-ACL*.
- Dowty, D. (1991). “Thematic Proto-Roles and Argument Selection.” *Language*, 67(3).
- Cruse, A. (2011). *Meaning in Language* (3rd ed.). Oxford University Press.
- Saeed, J. I. (2016). *Semantics* (4th ed.). Wiley Blackwell.
- Grice, H. P. (1975). “Logic and Conversation.” In *Syntax and Semantics 3*.
- Dagan, I., Glickman, O., & Magnini, B. (2006). “The PASCAL Recognising Textual Entailment Challenge.” In *Machine Learning Challenges*.

中文认知语义学著作可作为术语和理论脉络的辅助阅读。正式写入论文参考文献前，应
根据实际使用的纸质或电子版本核对作者、书名、版次、出版社和年份，不在本项目中
臆造页码。


# FastMTP 训练数据构造方案

## 1. 整体流程

```
源数据集 ──→ 提取 prompt x_n ──→ 主模型自蒸馏生成 ỹ_n ──→ 去重/清洗/混合 ──→ 389.4K 样本
```

严格按照论文 Section 2.4 的参数配置。

## 2. 源数据集

### 2.1 主体：Tulu 3 SFT Mixture (`allenai/tulu-3-sft-mixture`)

939,344 prompts，覆盖通用、数学、代码三大领域。仅提取 `messages[*].content` 中的 user prompt，**不使用原有 assistant 回复**。

| 子集 | 样本量 | 领域 | 用途 |
|------|------|------|------|
| FLAN v2 | 90K | 通用 + 阅读/总结 (→ RAG) | 通用 + RAG |
| WildChat GPT-4 | 100K | 通用对话 | 通用 |
| No Robots | 9.5K | 通用指令 | 通用 |
| OpenAssistant | 7K | 通用对话 | 通用 |
| CoCoNot | 11K | 上下文推理 (→ RAG) | RAG |
| Persona MATH | 150K | 数学推理 | 数学 |
| Persona GSM | 50K | 数学推理 | 数学 |
| Persona Algebra | 20K | 数学推理 | 数学 |
| NuminaMath-TIR | 64K | 数学推理 | 数学 |
| Persona Python | 35K | 代码 | 代码 |
| Evol CodeAlpaca | 107K | 代码 | 代码 |
| Persona IF | 30K | 通用指令 | 通用 |
| Aya | 100K | 多语言 (含中文) | 中文(部分) |
| WildGuardMix | 50K | 安全 | 可选 |
| 其他 | ~115K | — | 不采用 |

### 2.2 补充中文：BELLE 系列

论文要求 27% 中文。Tulu 3 中 Aya 中文占比低，需补充。

| 数据集 | 规模 | 用途 |
|------|------|------|
| `BELLE/train_3.5M_CN` | 3.5M (取子集) | 中文通用指令 |
| 或 `BAAI/CCI3-HQ` | — | 中文高质量对话 |

### 2.3 补充 RAG：外部数据集

Tulu 3 中 FLAN v2 和 CoCoNot 可作为 RAG 代理（含长上下文阅读/总结任务），但可额外补充。

| 数据集 | 规模 | 用途 |
|------|------|------|
| FLAN v2 (reading comp. 子集) | ~20K | RAG |
| CoCoNot | 11K | 上下文推理 |
| **`THUDM/LongBench`** | ~37K | 长文本 RAG（中英文，6 类任务，社区标准基准）|

## 3. 自蒸馏生成

### 参数（严格按论文）

```
temperature:   0.6
top-k:         20
top-p:         0.95
max_length:    4096
推理框架:      SGLang
```

### 注意

- 自蒸馏用主模型生成 response，确保 distribution alignment
- `temp=0.6` 引入多样性，同一个 prompt 可能生成不同 response
- 对于 predictor 训练，`temp=0.6` 的响应与 greedy target 不完全一致，会产生少量标签噪声。**这可以接受**——论文 389.4K 样本量足够平滑噪声，且 MTP 头本身也是在这个分布上训练的

## 4. 领域配比（调整后）

### 4.1 调整依据

基于我们 20+ 任务的 MTP 接受率评估结果：

| 领域 | Step1 典型接受率 | Step2+ 特点 | 对 predictor 的价值 |
|------|:---:|------|------|
| **代码** | 38-46% | Step2 > Step1，链延续性好 | 生成丰富正标签 |
| **RAG** | 55% | 各步接受率均匀，最高 | 最重要正标签来源 |
| **通用 QA** | 30-35% | 中等接受率 | 正负标签平衡 |
| **数学** | 10-22% | 接受率低，链易断 | 提供负标签，但有效标签少 |

### 4.2 建议配比

| 领域 | 占比 | 样本量 | 相比论文 | 调整原因 |
|------|:---:|------|:---:|------|
| **代码** | 25% | ~97K | +12% | Step2+ 正标签丰富，突破 K=3 的关键 |
| **RAG** | 20% | ~78K | 新增 | Step1 接受率最高 (55%)，predictor 正样本主来源 |
| **通用** | 25% | ~97K | -17% | 中等接受率，提供平衡样本 |
| **数学** | 10% | ~39K | -8% | 接受率低，有效标签少，降低占比避免负标签主导 |
| **中文** | 20% | ~78K | -7% | 为代码/RAG 腾出空间 |
| **合计** | **100%** | **389.4K** | — | — |

### 4.3 采样策略

从源数据集按比例分层采样。若某个领域不够，可：
- 代码：Tulu 3 CodeAlpaca (107K) + Persona Python (35K) = 142K → 取 97K ✓
- RAG：FLAN v2 reading comp. (~30K) + CoCoNot (11K) + LongBench (~37K) → 取 78K
- 通用：FLAN v2 通用子集 + WildChat + No Robots → 取 97K ✓
- 数学：Persona MATH/GSM/Algebra (220K) → 取 39K ✓
- 中文：BELLE 子集 + Aya 中文部分 → 取 78K

## 5. 数据清洗

按论文流程：

```
1. MinHash 全局去重（源数据集内 + 跨数据源）
2. 启发式过滤：
   - 不完整/截断的推理链
   - 过度重复内容
   - 长度不在期望范围内
3. 数据混合：按 4.2 配比加权
```

## 6. 输出格式

标准 SFT JSONL，兼容 ms-swift：

```jsonl
{"messages": [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<self_distilled_response>"}]}
```

**不需要额外元数据**。Predictor 训练时的标签在 forward 过程中动态生成（比较 MTP draft 与 answer token），不依赖数据中的领域标注。

## 7. 实现步骤概览

```
Step 1: 下载 Tulu 3 SFT Mixture + BELLE + RAG 数据集
Step 2: 提取所有 user prompt，按领域分类
Step 3: 按 4.2 配比分层采样 → prompt pool
Step 4: SGLang 自蒸馏生成 responses (temp=0.6, top-k=20, top-p=0.95)
Step 5: MinHash 去重 + 启发式清洗
Step 6: 输出标准 SFT JSONL 格式 → 训练就绪
```

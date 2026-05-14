# FastMTP Adaptive K Predictor — 训练方案

## 1. 背景与动机

FastMTP 使用 Looped LM（单 MTP 头，迭代 K 次）进行推测解码。各任务 MTP 接受率差异巨大：

| 任务 | Step1 Acc | Step2 Acc | Step3 Acc | K=3 EAL |
|------|:---:|:---:|:---:|:---:|
| math/Prealgebra | 62.5% | 12.5% | 0% | 1.625 |
| spec_bench/rag | 55.0% | 50.0% | 21.7% | 1.883 |
| livecodebench/hard | 46.4% | 60.7% | 42.9% | 1.929 |
| mt_bench/math | 10.0% | 30.0% | 10.0% | 1.100 |

**核心思路**：引入轻量 predictor，在 MTP 迭代过程中动态决定继续/停止：
- 对高接受率任务/样本 → 多步草稿（可超过训练时的 K=3）
- 对低接受率任务/样本 → 提前止步

---

## 2. Predictor 架构

```
StopPredictor:
  设计原则：用 MTP 第 k+1 步的输入（而非第 k 步的输出）预测"第 k+1 步是否值得继续"。
  input_to_step_k+1 = input_proj(concat(hidden_layernorm(hidden_k), token_layernorm(emb(0))))
  已在 MTP 训练中使用，融合了"上步上下文"和"当前 token 嵌入"。

  Input:
    - next_step_input  : MTP 第 k+1 步的投影输入 [4096] (detached)
                         = input_proj(concat(hidden_layernorm(hidden_k), token_layernorm(emb(0))))
    - confidence_k     : 第 k 步 softmax 最高概率 [1]
    - entropy_k        : 第 k 步预测分布熵 [1]
    - step_embedding   : 第 k+1 步索引 {2,3,4,5} → learnable embedding [16]

  Architecture:
    Linear(4096 + 1 + 1 + 16 → 1) + Sigmoid

  Output:
    P(accept_{k+1} | 当前状态) ∈ [0, 1]
    含义：在当前 MTP 状态下，下一步的草案被验证接受的概率。
    注意：p 是对"下步是否会被接受"的预测，不是对"当前步已经被接受"的判断。

  Params: ~4.1K (=16KB)
  Cost:   可忽略（input_proj 是 MTP 层已有的 Linear）
```

### emb(0) 来源

`modeling_mimo.py` L19-26 中 `roll_tensor(input_ids, shifts=-1, fill_num=0.0)` 将回滚位置填为 token 0，随后 `embed_tokens(input_ids)` 取其嵌入。MTP 已在此设定下训练完毕（K=3），predictor 直接复用 `emb(0)`，与 MTP 训练保持一致。

---

## 3. 训练标签生成

### 3.1 核心原则

- **标签来源**：自蒸馏训练数据的 `answer_ids`。数据已被视为正常 SFT 数据，MTP 头本身也在这份数据上训练。用同一份数据保证一致性。
- **条件标签**：Step k 有效标签仅在 Step 1..k-1 **全部正确**时存在（推测解码链中的条件依赖）
- **Teacher-forcing**：单次 `forward(prompt + answer)` 获得所有 MTP 步骤的预测和隐状态

### 3.2 标签生成算法

```
输入: prompt_ids, answer_ids
      full_seq = prompt_ids + answer_ids
      L = len(prompt_ids)

前向:
  outputs = model(full_seq)
  # outputs.hidden_states_mtp[k] — MTP 第 k+1 步隐状态（沿模型自身 looped trajectory 迭代）
  # 提取位置 L-1（最后一个 prompt token）的预测

链模拟（沿 MTP 自身轨迹）:
  chain_alive = True
  labels = {}

  for step in 1..K:
      if not chain_alive:
          labels[step] = -100     # IGNORE（推测解码中不会到达）
          continue

      draft_k = argmax(lm_head(outputs.hidden_states_mtp[step-1][:, L-1, :]))
      target_k = answer_ids[step]   # 自蒸馏训练数据的第 step 个 token

      if draft_k == target_k:
          labels[step] = 1         # 草案被接受（与训练数据匹配）
          # chain 保持存活
      else:
          labels[step] = 0         # 草案被拒绝
          chain_alive = False      # 链断裂，后续步骤全部 IGNORE
```

### 3.3 对齐逻辑

`modeling_mimo.py` L239-253 中 MTP 第 k 步计算 loss 时，`labels` 被 `roll_tensor(labels, shifts=-1, fill_num=-100)` 回滚 k 次。因此 MTP 第 k 步在位置 L-1 预测的是 `answer_ids[step]`（即位置 L+step 的 token）。

| 来源 | 位置 | 预测目标 |
|------|------|----------|
| `lm_head(main_hidden[:, L-1, :])` | L-1 | `answer[0]` (t_0) |
| `lm_head(mtp_step1[:, L-1, :])` | L-1 | `answer[1]` (t_1) |
| `lm_head(mtp_step2[:, L-1, :])` | L-1 | `answer[2]` (t_2) |
| ... | L-1 | ... |

### 3.4 标签分布特征

- Step 1：全部样本有有效标签
- Step 2：仅 Step 1 成功的样本有标签
- Step 3：仅 Step 1+2 都成功的样本有标签
- 正负样本不平衡：后期步骤正样本少、IGNORE 多

---

## 4. 训练流程

### 4.1 分阶段训练

我们已有训好的 K=3 MTP 模型（`TencentBAC/FastMTP`）。采用分阶段策略：

```
阶段 1 (已完成): 训练 MTP → K=3 模型就绪
阶段 2 (当前):   冻结 MTP，仅训练 predictor
```

**理由**：MTP 表示稳定后，predictor 学到"稳态 MTP 特征 → 是否该继续"的映射，而不是追着移动的 MTP 表示跑。

### 4.2 训练伪代码

```
for batch in dataloader:
    prompt_ids, answer_ids = batch
    L = prompt_ids.shape[1]
    full_seq = torch.cat([prompt_ids, answer_ids], dim=1)

    # ====== 1. MTP 前向（no grad for MTP params） ======
    with torch.no_grad():
        outputs = model(full_seq)
    # 注: MTP 已冻结，不需要计算 MTP CE loss

    # ====== 2. 条件标签生成 ======
    chain_alive = True
    loss_pred_total = 0.0

    for step in 1..K:
        if not chain_alive:
            continue

        logit_k = lm_head(outputs.hidden_states_mtp[step-1][:, L-1, :])
        draft_k = argmax(logit_k)
        target_k = answer_ids[:, step]
        label = (draft_k == target_k).float()   # 1 or 0

        # ====== 3. Predictor 输入构造 ======
        hidden_prev = outputs.hidden_states_mtp[step-1][:, L-1, :]
        emb_0 = model.model.embed_tokens(torch.tensor([0], device=device))

        next_input = model.model.mtp_layers[0].input_proj(
            torch.cat([
                model.model.mtp_layers[0].hidden_layernorm(hidden_prev),
                model.model.mtp_layers[0].token_layernorm(emb_0)
            ], dim=-1)
        )

        conf_k = softmax(logit_k).max()
        ent_k = -(softmax(logit_k) * log(softmax(logit_k))).sum()

        # ====== 4. Predictor 预测 ======
        p_logit = predictor(next_input, conf_k, ent_k, step_emb(step + 1))
        loss_pred_total += BCEWithLogitsLoss(p_logit, label, pos_weight=pos_w)

        chain_alive = (label == 1)

    # ====== 5. 仅更新 predictor ======
    loss_pred = loss_pred_total / valid_step_count
    loss_pred.backward()   # 梯度仅流向 predictor（MTP 在 no_grad 中）
    optimizer.step()
```

### 4.3 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 前向次数 | **单次** | full_seq 含 answer，teacher-forcing |
| 标签来源 | 自蒸馏训练数据的 answer | 与 MTP 训练数据一致 |
| 训练策略 | **阶段 2：冻结 MTP，仅训 predictor** | MTP 已收敛，predictor 学稳态映射 |
| 梯度流向 | 仅更新 predictor | MTP 在 `torch.no_grad()` 中 |
| 损失函数 | **BCEWithLogitsLoss + pos_weight** | 概率校准重要（推理要扫阈值） |
| IGNORE 标签 | 跳过 loss 计算 | `label == -100` 不参与 |
| emb(0) | 复用 `model.embed_tokens(0)` | 与 MTP 训练一致 |
| 阈值策略 | **单全局阈值 + step_embedding** | predictor 通过 step_embedding 自适应不同步数，阈值统一 |

### 4.4 训练参数

```
pos_weight:  neg_count / pos_count（自动计算，平衡正负样本）
学习率:      1e-3 ~ 5e-4
Epochs:      3 ~ 5
Optimizer:   AdamW
Batch size:  与 MTP 训练一致
```

---

## 5. 推理时部署

```python
def generate_with_adaptive_k(model, predictor, prompt, max_k=5, threshold=0.5):
    """
    threshold 控制激进程度: 低 = 激进 (容易继续), 高 = 保守 (容易停)
    p 的含义: P(下步草案会被 verifier 接受 | 当前 MTP 状态)
    """
    # Step 0: 主模型预测 t_0
    outputs = model(prompt)
    t_0 = argmax(outputs.main_logits)
    hidden = outputs.hidden_states_main
    drafts = [t_0]

    for step in 1..max_k:
        # MTP 迭代一步
        next_input = model.mtp_layers[0].input_proj(
            cat([hidden_layernorm(hidden[:, -1, :]),
                 token_layernorm(emb(0))], dim=-1)
        )
        hidden = model.mtp_layers[0](emb(0), hidden, ...)
        logit = model.lm_head(hidden[:, -1, :])
        draft = argmax(logit)
        conf = softmax(logit).max()
        ent = -sum(softmax(logit) * log(softmax(logit)))

        # predictor 判断: 下步 (step+1) 是否值得继续
        p = predictor(next_input.detach(), conf, ent, step + 1)
        if p < threshold:
            break

        drafts.append(draft)

    return drafts  # → target model 验证
```

阈值选择：验证集上扫描 `[0.1, 0.9]`，选 EAL vs mean_K 的 Pareto 前沿。

---

## 6. 整体架构

```
训练时 (阶段 2):
  prompt + answer ──→ model.forward() [no_grad]
      │                   │
      │                   ├──→ hidden_states_mtp (detached)
      │                   │              │
      │                   │              ├──→ next_step_input (emb(0) + proj)
      │                   │              ├──→ confidence + entropy
      │                   │              └──→ step_embedding
      │                   │                        │
      │                   └──→ 条件标签 (draft vs answer) ──┘
      │                                                      │
      │                        BCEWithLogitsLoss(P(accept), label)
      │                                                      │
      │                        loss_pred.backward() → predictor only

推理时:
  prompt → base model → t_0
              │
              └──→ MTP step 1 → predictor → P(step2_OK) > θ?
                      │                    yes → step 2 → ...
                      └──→ draft_1         no  → stop
```

---

## 7. 预期效果

| 场景 | 固定 K=3 | 自适应 K | 预期改进 |
|------|:---:|:---:|:---:|
| livecodebench (高接受率) | K=3, EAL=1.93 | K=4~5 | EAL → 2.2+ |
| mt_bench/math (低接受率) | K=3, EAL=1.10 | K=0~1 | 少浪费 |
| spec_bench/rag (中等) | K=3, EAL=1.88 | K=2~4 (per-sample) | 效率提升 |

---

## 8. 实现步骤

1. **数据准备**：按 `training_data_plan.md` 构造自蒸馏 SFT 数据
2. **Predictor 模块**：在 `modeling_mimo.py` 添加 `StopPredictor` 类
3. **训练脚本**：阶段 2 训练（冻结 MTP，仅训 predictor）
4. **验证**：在 eval 数据集上扫阈值，测 EAL vs mean_K
5. **集成推理**：修改 SGLang EAGLE 推理代码

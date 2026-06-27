# SOAP-Core Current Status

- 日期：2026-06-28
- HEAD：`5b52128`（v0.7.3 docs commit，已 push origin/main）
- 工作区：**v0.7.4 代码 + 实验 + 文档已完成，未 commit**（待用户决定，只 diff 不 push）
- 本文不含任何 key / token / base_url。

---

## 1. 项目出发点与核心问题

- **定位**：领域无关的复杂系统状态空间解释层。核心不绑场景，中医/AI训练/工业/金融都是应用层。
- **核心问题**：训练失稳 ≠ 变随机，而是状态空间结构变化。SOAP 提取结构指纹并分类失稳。
- **首选场景**：AI training instability 预警。

## 2. 理论逻辑链

原始多变量时序 → 标准化/窗口化 → 有效维度 d*（PCA + train/test scoring）→ 相空间重构 → 低维投影 → 吸引子聚类 → 状态转移 → Simplex/S-Map 预测 → RQA → 结构指纹 → 应用层。

## 3. 两层诊断架构（不变）

- **层 1 SOAP dynamics**：divergence 有效；mode_collapse 不敏感。
- **层 2 Representation geometry**：mode_collapse 有效（effective_rank/collapse_score）。
- **分层原则**：动力学失稳用 SOAP 状态空间；representation 失稳用几何指标直判。两层互补。

## 4. v0.7.4 已完成：HuggingFace 多层 representation collapse benchmark

详细见 `docs/v0.7.4_hf_multilayer_benchmark.md`。要点：

### 实验设计
- 模型 `distilbert-base-uncased`（冻结 embedding + 前 3 block，可训练 block 3-5 + 分类头 mask 加权 meanPool + Linear(768,5)）
- 数据 5 类结构化英文模板（train 1600 / val 400 / probe 128），`data_seed=0` 固定，seed 仅控模型/dropout/batch 顺序
- 训练 120 步 / batch 16 / AdamW lr 2e-5 wd 0 / 每 3 步在固定 probe set 记录（40 checkpoint）；seed 42/43/44
- normal vs induced collapse（**controlled progressive rank-1 activation intervention**）

### Controlled rank-1 intervention（人工干预，非自然 collapse）
- 在 block 3 输出（`hidden_states[4]`）于 **sample pooled space** 构造 rank-1 目标，写回完整 hidden states（`masked_mean(modified)==target`）
- `target = global_center + (1-α)·centered + α·((centered·u)·u)`；u = 初始 probe pooled 的 PCA 第一主方向（全 run 固定）
- α schedule：step 0-20 = 0；step 21-120 线性 → 0.98；只保留 classification loss

### 方法论负面结果（4 轮失败，保留不隐藏）
共同根因：**effective_rank 是归一化能量集中度，干预须让能量集中到少数方向且与检测指标同空间（sample pooled）**。
1. `-log(var)` 方向反（最小化让 var 增大）
2. token-level `log(var)`：度量空间不一致
3. sample-level `log(var)`：isotropic 均匀缩，归一化 effective_rank 反增
4. rank-1 token-centered（每样本 token mean）：保留样本间满秩差异

### 校准：leave-one-seed-out
2 个 normal seed 校准逐层 collapse_score 99 分位阈值，1 个 normal seed 评估 FPR；阈值**只用 normal**，禁用 induced 反向调参。双规则 any_layer + persistent（同层连续 ≥3 checkpoint）。

### 结果
- induced **3/3 检出**（any_layer + persistent），首现层 `layer_4 @ step_102`，传播 `layer_4→5→6` 三 seed 一致
- `layer_4` induced collapse_score 0.974 vs 阈值 0.106（**9 倍区分**，rank-1）
- normal FPR（**3 seed 小样本，0.333 = 1/3 held-out seed 误报，不外推**；预训练 DistilBERT 合成任务 normal run）：冻结层 `layer_0-3` = 0；可训练深层 `layer_4/5/6` per-layer FPR = 1/3；run-level any_layer FPR = persistent FPR = 1/3

### 双面结论
1. controlled collapse 可有效检测（3/3，首现层 = 施加层，传播一致，layer_4 9 倍区分）
2. **层数增加暴露误报**（v0.7.3 toy 2 层未暴露，**仅限本 controlled benchmark 设置**）：预训练 DistilBERT 合成任务 normal run 的可训练深层后期 effective_rank 自然下降，collapse_score 持续超阈值 → 深层 per-layer FPR 1/3（小样本不外推）；persistent 规则未降误报（持续低秩倾向，非瞬时波动）。**结论限于 controlled rank-1 benchmark，不代表自然 mode collapse 已验证**
3. 冻结层 FPR 0 证实干预空间特异性
4. collapse 未完整传播（block 4/5 可训练层具表示重扩张能力）

## 5. task 33-39 状态

| task | 状态 |
|---|---|
| 33 transformers+distilbert | ✅ |
| 34 hf_repr_experiment.py | ✅（方案5 pooled rank-1，pilot layer_4→1.03） |
| 35 hf_collapse_calibration.py | ✅ |
| 36 正式 6 runs（3 seed × normal/induced） | ✅ |
| 37 分析误报率/传播路径 | ✅ |
| 38 v0.7.4 benchmark 文档 | ✅ |
| 39 memory + diff（不 push） | 🔄 进行中 |

## 6. 关键文件

- `soap/apps/training/hf_repr_experiment.py`（实验脚本，rank-1 intervention）
- `soap/apps/training/hf_collapse_calibration.py`（leave-one-seed-out 校准 + 双规则 + 传播）
- `docs/v0.7.4_hf_multilayer_benchmark.md`（benchmark 科学文档，含 4 失败负面结果）
- `docs/v0.7.4_calibration_report.md`（校准报告）
- `docs/v0.7.4_experiment_spec.md`（实验设计 spec）
- `examples/hf_repr_seed{42,43,44}_{normal,condition_b}.csv`（6 runs 正式数据）
- `examples/hf_pilot/`（5 轮 pilot，含 4 失败机制负面结果，保留）

## 7. 下一步

- 待用户决定 v0.7.4 commit（**只 diff 不 push** 已执行）
- **v0.7.5（下一步，先于 v0.8）**：深层误报抑制 —— 层级基线校准 / 按层动态阈值 / 区分"训练后期自然聚焦"与 controlled collapse（深层 per-layer FPR 1/3 待降）。**统一 CLI 推迟到 v0.8**（深层 per-layer FPR 1/3 说明先做 v0.7.5 校准再封装）

## 8. 管家模式产出

- MiMo 写 hf_repr_experiment.py（34A/B/C 分阶段 + max_tokens 16384）+ hf_collapse_calibration.py + v0.7.4 文档；主线程审核 + 真跑验证 + 机制设计迭代
- 关键修复：mimo_worker `trust_env=False`（macOS 系统代理致 ProxyError，根治）
- 5 轮 collapse 机制迭代（4 失败 + 方案5 成功），真跑驱动设计纠偏

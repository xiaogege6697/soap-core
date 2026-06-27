# SOAP-Core Current Status

- 日期：2026-06-27
- HEAD：`abe58b4`（v0.7.3，已 push 到 origin/main）
- 工作区：干净（v0.7.4 阻塞未产出）
- 本文不含任何 key / token / base_url（遵持久化收尾要求）。

---

## 1. 项目出发点与核心问题

- **定位**：领域无关的复杂系统状态空间解释层。从多变量时序中提取结构指纹（最优有效维度、重构状态空间、吸引子、状态转移、预测风险路径），用于失稳检测、分类与早期预警。核心不绑场景，中医/AI训练/工业/金融都是应用层。
- **核心问题**：训练失稳 ≠ 变随机，而是状态空间结构变化。SOAP 能否提取结构指纹并分类失稳？
- **首选场景**：AI training instability 预警（loss/grad/lr/val 天然存在，决策闭环短：暂停/降 lr/回滚）。

## 2. 理论逻辑链

```
原始多变量时序
  → 标准化 / 窗口化
  → 有效维度搜索 d*（variance / PCA + train/test scoring）
  → 相空间重构（window 或 Takens + AMI + FNN）
  → 低维投影
  → 吸引子聚类（k-means）
  → 状态转移矩阵 + next-state 概率
  → 预测诊断（Simplex / S-Map）
  → recurrence 诊断（RQA：recurrence_rate / determinism / 对角线）
  → 结构指纹
  → 应用层（失稳检测 / 分类 / 预警）
```

## 3. 当前两层架构及各自边界

### 层 1：SOAP dynamics layer（动力学失稳）
- **方法**：d*（PCA）+ attractor/transition + simplex/smap + recurrence（RQA）。
- **有效**：divergence（skill 崩 / det 骤降 / 吸引子变）、overfit（加 train_val_gap 后 d* 扩张）。
- **边界**：对 mode_collapse 等业务语义失稳不敏感——曲线平滑可预测，SOAP 指纹近 normal。

### 层 2：Representation geometry layer（表示失稳，direct detection）
- **方法**：隐藏层 effective_rank（entropy）/ representation_variance / collapse_score（=1/eff_rank，proxy）。
- **有效**：mode_collapse（collapse_score 高 / eff_rank 近 1）。
- **边界**：toy 阈值（collapse_score>0.9 且 eff_rank<1.5）benchmark-specific，真实模型需按层按任务校准；collapse_score 是 proxy 非严格度量（真实可用 NC1/CCA 等）。

### 分层原则
**动力学失稳用 SOAP 状态空间；representation 失稳用几何指标直判**（不强行塞进 SOAP pipeline）。两层互补：divergence 在两层都触发，mode_collapse 只在 representation 层触发，normal/overfit 两层都不触发。

## 4. v0.6.8 → v0.7.3 的证据演进与纠偏

| 版本 | 证据 | 纠偏 |
|---|---|---|
| v0.6.8 | 真实 PyTorch 训练：taxonomy 部分迁移（divergence 可识别，overfit/mc 不可分） | v0.6.4 synthetic taxonomy **不可直接外推**；realish "完美迁移"是假象（本质 synthetic 套壳） |
| v0.6.9 | 加 train_val_gap + output_variance：overfit 部分解锁（d* 四类各不同），mc 仍弱 | **标量观测层触顶**——mc 需 representation features |
| v0.7 | representation geometry：mc 被明显拉开，但 SOAP 动力学指纹仍不分 | 确立**失稳检测分层** |
| v0.7.1 | adapter direct detection；规则 B（drop_ratio）误判 normal/overfit 已移除 | collapse 以 **final 值**为准，不以变化比例为准（eff_rank 训练中自然下降） |
| v0.7.2 | representation input schema 定义 | toy 阈值**不固化进 CLI** |
| v0.7.3 | 多层：mc 首现深层 hidden_1；any_layer 聚合 4/4 正确 | collapse 是**局部深层**现象；全层聚合（均值/连续）掩盖局部 |

**关键纠偏**：any_layer 的 4/4 是 toy（2 层）结果，**层数增加后多重比较会抬升误报率**（v0.7.4 待验证）；consecutive/severity_weighted 在 toy 漏判。

## 5. v0.7.4 实验目标、模型选择、校准原则

- **目标**：真实多层 transformer 验证 representation collapse 的**逐层阈值、误报率、坍缩传播路径**（不只是接 HuggingFace）。
- **模型**：distilbert-base-uncased（6 层，hidden=768，66M）—— 已下载就绪。
- **条件**：
  - normal：微调，weight_decay=0（校准用）。
  - induced collapse：微调，weight_decay=1.0 强正则干预（**文档须说明干预机制，不冒充自然训练坍缩**）。
- **seed**：42 / 43 / 44（≥3）。
- **校准原则（严格）**：**阈值仅用 normal calibration 数据确定**（逐层 collapse_score 99 分位），**不能用 collapse 数据反向调参**。
- **双规则**：any_layer（任一层单次命中）+ persistent（同一层连续 ≥3 checkpoint 命中）。
- **传播路径**：记录 collapse 首现层 + 向后续层传播顺序。
- **验收**：阈值仅 normal 校准 / ≥3 seed / induced 干预说明 / 保留负面结果 / **只 diff 不 push**。

## 6. task 33–39 当前状态与阻塞

| task | 状态 |
|---|---|
| 33 装 transformers + distilbert | ✅ 完成（6 层 768 就绪） |
| 34 写 hf_repr_experiment.py | ⏸ 暂停（阻塞） |
| 35 写校准 + persistent 聚合 | pending |
| 36 跑实验（3 seed） | pending |
| 37 分析误报率 / 传播路径 | pending |
| 38 写 v0.7.4 文档 | pending |
| 39 更新 memory + 出 diff（不 push） | pending |

**阻塞**：辅助代码/文档生成模型（MiMo）接入待恢复。恢复后，task 34（hf_repr_experiment.py）spec 已完整设计（见 `memory/projects/soap-core.md` v0.7.4 段），立即重派即可继续。

## 7. 下一步准确执行顺序

1. 恢复 MiMo 接入（用户配置，不写明文 key 到源文件——用环境变量）。
2. 派 MiMo 写 `hf_repr_experiment.py`（spec 已备）。
3. 主线程跑实验（3 seed × normal/induced，逐层时序 CSV）。
4. 派 MiMo 写 `hf_collapse_calibration.py`（仅 normal 校准 + any_layer/persistent + 传播路径）。
5. 主线程分析逐层误报率与传播顺序。
6. 派 MiMo 写 v0.7.4 benchmark 文档。
7. 主线程更新 memory + PROJECT_OVERVIEW，**只 diff 不 push**。
8. 之后：v0.8 统一 CLI / report（基于 any_layer 聚合 + 逐层诊断，封装两层诊断）。

## 8. 管家模式及审核教训

**分工**：MiMo 写代码/文档（省智谱 glm），主线程规划 + 审核 + 真跑验证（不可外包）。

**主线程审核抓出的关键问题**：
- v0.6.8 路径 bug（`parents[2]`→`[3]`，spec 写错）；divergence 动力学不达标（full-batch 无噪声→改 mini-batch SGD）。
- v0.7.1 adapter 规则 B 误判（drop_ratio 移除，改 final 值判定）。
- MiMo 产出围栏残留（多次）、文档指令语残留（2 次，已沉淀 lesson）。

**沉淀的教训**（见 memory）：
- MiMo 超时优先重试等待，不主线程接手（除非确认不可恢复）；key 失效用环境变量方案。
- 给 MiMo 写文档 prompt 避免"请明确写出/必须写出"（会被原样抄进正文）。
- v0.6.4 教训：数据语义 bug 扭曲 d*（grad_norm 负值），合成与真实数据都须先校验字段语义。

**关键文件**：
- `docs/PROJECT_OVERVIEW.md`（架构 + 版本表，第 6 节两层诊断）
- `docs/v0.6_*` / `docs/v0.7_*` benchmark 文档（各版本证据链）
- `~/.claude/projects/-Users-xiaogege9967/memory/projects/soap-core.md`（版本线 + v0.7.4 进行中状态）
- `~/.claude/projects/-Users-xiaogege9967/memory/facts/mimo-worker.md`（MiMo 用法 + key 切换）

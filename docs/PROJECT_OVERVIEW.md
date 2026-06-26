# SOAP-Core — Scale-Optimized Attractor Prediction Core

> 一句话定位：**领域无关的复杂系统状态空间解释层**。

SOAP-Core 从多变量时间序列中提取系统的**结构指纹**——最优有效维度、重构状态空间、吸引子、状态转移与预测风险路径——用于失稳检测、分类与早期预警。它不绑定中医、AI 训练、金融、气候；这些都只是应用层。

---

## 1. 出发点

现有生态分裂为两类，中间层空缺：

- **学术方法库**（pyEDM / DynamicalSystems.jl / DelayEmbeddings.jl / DADApy / pyunicorn / giotto-tda）：算法强，但产品统一性弱，使用者要自己拼装。
- **行业产品**（W&B / TensorBoard）：场景强，但偏 dashboard 与阈值告警，缺少"状态空间结构解释"。

SOAP-Core 占住中间层：把动力系统、EDM、TDA、intrinsic dimension、early warning signals 等方法，产品化成一个**通用状态空间解释引擎**。

**核心主张**：预测的不是未来精确点位，而是——吸引子归属、状态转移概率、临界风险、低维演化趋势、干预后可能路径。

---

## 2. 理论基础

| 支柱 | 作用 |
|---|---|
| 动力系统与吸引子 | 系统长期演化收敛到的低维结构，是状态空间的"骨架" |
| Takens delay embedding 定理 | 从单/多变量时间序列重构相空间（观测→状态） |
| AMI（平均互信息） | 估计最优 delay τ |
| FNN（虚假近邻） | 估计 embedding dimension m |
| Intrinsic Dimension / 有效维度 d\* | 状态空间最小有效表达维度（PCA 路径） |
| EDM（经验动力学建模） | Simplex（近邻加权预测）、S-Map（局部线性预测） |
| Recurrence Quantification Analysis | recurrence rate、determinism、对角线结构 |
| 吸引子状态转移 | k-means 候选聚类 + 转移矩阵 + next-state 概率 |
| Early warning signals | 结构变化作为临界转变前兆 |

---

## 3. 理论逻辑脉络

```
原始多变量时序
  → 标准化 / 窗口化
  → 有效维度搜索 d*（variance / PCA + train/test scoring）
  → 相空间重构（window 或 Takens + AMI + FNN）
  → 低维投影
  → 吸引子聚类（k-means）
  → 状态转移矩阵 + next-state 概率
  → 预测诊断（Simplex / S-Map）
  → recurrence 诊断（RQA）
  → 结构指纹
  → 应用层（失稳检测 / 分类 / 预警）
```

---

## 4. 架构实现蓝图

```
soap/
├── data/          loader（CSV→TimeSeries）、preprocessing（standardize / sliding_windows）
├── dimension/     selector（variance d*）、pca_selector（PCA d* + train/test scoring）
├── embedding/     takens（delay embedding）
├── metrics/       ami（delay 估计）、fnn（embedding dim 估计）、recurrence（RQA）
├── attractor/     clustering（k-means 候选）
├── prediction/    transition（转移矩阵）、simplex、smap
├── report/        generator（markdown report + SVG）
├── apps/training/ generator（合成训练日志）、adapter（load_training_run）、real_adapter（v0.6.5 真实日志）
└── cli.py         analyze / generate-example / generate-lorenz 命令入口
```

每层职责单一、可独立替换；核心不绑场景，应用层（`apps/`）按领域接入。

---

## 5. 版本演进

| 版本 | 能力 |
|---|---|
| v0.3 | PCA effective dimension + 时间顺序 train/test predictive scoring |
| v0.4 | Takens delay embedding + AMI/FNN 自动参数估计 |
| v0.5 | Simplex / S-Map prediction diagnostics + recurrence diagnostics（实现并接入） |
| v0.6 | 应用适配层 `soap/apps/training/`，首个场景 AI training instability |
| v0.6.1~v0.6.4 | training benchmark 建立**失稳 taxonomy**（见下） |
| v0.6.5 | 真实训练日志统一 schema + field_map + 字段语义校验 |

---

## 6. 核心产品假设（v0.6.4 验证）

**训练失稳 ≠ 变随机，而是状态空间结构变化。** SOAP 能提取结构指纹并分类失稳：

| 类型 | d\* | determinism | grad/loss 行为 |
|---|---|---|---|
| normal（基线） | 2 | 0.52 | 稳定 |
| **divergence 型** | 1 ↓ | 0.96 ↑ | grad 爆炸 + loss 爆炸 |
| **mode_collapse 型** | 1 ↓ | 0.86 ↑ | grad 衰减 + loss 平台 |
| **overfit / 扩张型** | 3 ↑ | 0.59 | train 降 val 升 |

- **最强组合信号**：d\* 方向（升/降）+ determinism 是否飙升 + grad/loss 行为。
- **二级分类**：坍缩型失稳内部，由 grad/loss 方向区分 divergence（爆炸）vs collapse（衰减）。
- prediction skill 不能单独作风险指标（diverging 反而最高）。
- **边界**：当前 taxonomy 基于合成数据，真实训练日志验证前不过度外推。

---

## 7. 应用场景

- **首选：AI training instability 预警**——loss / grad_norm / lr / val_loss 天然存在，决策闭环短（暂停 / 降 lr / 回滚 checkpoint），竞品偏阈值告警、少状态空间解释。
- **备选**：工业 predictive maintenance、金融 / 气候临界转变、生理信号、中医状态辨证（领域无关底座的垂直应用）。

---

## 8. 设计哲学

- **领域无关底座**：应用层接入，核心不绑场景。
- **最小依赖**：核心可纯标准库启动，科学计算路径可选（numpy / scikit-learn）。
- **CLI 驱动 + 合成数据验证**：每个版本有 smoke benchmark。
- **数据语义严格**：字段校验是分析前提（grad_norm 负值会扭曲 d\*——v0.6.4 教训）。

---

## 9. 路线图

- **v0.6.6**：真实 / 半真实 training CSV benchmark，验证 synthetic taxonomy 是否迁移。
- **后续**：takens embedding 对照、自动分类规则、W&B API / TensorBoard event 直连、其他应用场景（工业 / 金融）。

---

## 快速运行

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[science]"
.venv/bin/python -m soap.cli generate-example --output examples/synthetic_cyclic.csv
.venv/bin/python -m soap.cli analyze examples/synthetic_cyclic.csv --embedding window --method pca --predictor simplex --recurrence --clusters 3
```

详细 CLI 参数与各版本实现笔记见 `docs/`。

# SOAP-Core Current Status

- 日期：2026-06-28
- HEAD：`fcc39f2`（v0.7.4 已 push origin/main）
- 工作区：**v0.7.5 代码 + 实验 + 文档已完成，未 commit**（待 audit + commit，不 push）
- 本文不含任何 key / token / base_url。

---

## 1. 项目出发点与核心问题

- **定位**：领域无关的复杂系统状态空间解释层。核心不绑场景，中医/AI训练/工业/金融都是应用层。
- **核心问题**：训练失稳 ≠ 变随机，而是状态空间结构变化。
- **首选场景**：AI training instability 预警。

## 2. 理论逻辑链

原始多变量时序 → 标准化/窗口化 → 有效维度 d*（PCA + train/test scoring）→ 相空间重构 → 低维投影 → 吸引子聚类 → 状态转移 → Simplex/S-Map 预测 → RQA → 结构指纹 → 应用层。

## 3. 两层诊断架构（不变）

- **层 1 SOAP dynamics**：divergence 有效；mode_collapse 不敏感。
- **层 2 Representation geometry**：mode_collapse 有效（effective_rank/collapse_score），**phase-aware baseline deviation detection**（v0.7.5）。
- **分层原则**：动力学失稳用 SOAP；representation 失稳用几何指标直判。两层互补。

## 4. v0.7.4 已 push（commit fcc39f2）

预训练 DistilBERT 合成任务 controlled rank-1 collapse benchmark。induced 3/3 检出（layer_4 首现，传播 4→5→6）；static per-layer threshold 深层 normal per-layer FPR 1/3（3 seed 小样本）。详见 docs/v0.7.4_hf_multilayer_benchmark.md。

## 5. v0.7.5 已完成（工作区，未 commit）：phase-aware baseline deviation detection

详细见 `docs/v0.7.5_phase_aware_benchmark.md`。

### 关键认知校正
- 自然训练后期有**良性 representation concentration**（Neural Collapse, Papyan et al.），秩下降≠故障。
- checkpoint 非独立，**不宣称 conformal 严格 FPR**，只借鉴 CADE 多层组合。
- 只称 **baseline deviation detection**，**不宣称**区分良性 NC vs 病态 collapse。

### 三方法 A/B/C（held normal 20 / held induced 3，dev normal 42-53 校准）
- **A** static（v0.7.4）：FA **20/20**（Wilson 0.84-1.00），Det 3/3。当前 DistilBERT synthetic-task 下深层 normal 后期完全失效。
- **B** phase-knots（8 knots 线性插值 median/MAD + MAD layer floor + run-level max anomaly，threshold_B=10.79 @q0.95）：FA **0/20**（Wilson 0.00-0.16），Det 3/3。
- **C** B+persistent：FA **0/20**，Det 3/3（与 B 同，persistent 无增益）。
- Exact McNemar：A vs B/C n10=20 p=0.000002（B/C 显著优 A）；B vs C p=1.0。
- 两 q（0.95 主 / 0.90 sensitivity）结果一致。

### 产品结论
- **推荐 phase-aware B 为默认**，替代 static A。
- **persistent C 暂不进默认路径**（无增益，避免复杂度）。
- 完美分离（0/20 vs 3/3）部分源于 controlled rank-1 intervention 与 effective-rank 数学对齐，**不归因于对自然 collapse 的判别力**。

### 边界（不外推）
- controlled benchmark / 预训练 DistilBERT 合成任务；跨任务/跨模型/自然 mode collapse **未验证**。
- 小样本（20/3）Wilson CI，0/20 CI 上界 0.16；phase baseline 须按模型/层/任务**重新校准**，threshold_B=10.79 不可迁移。

## 6. task 状态

v0.7.4 task 33-39 ✅（已 push）。v0.7.5 task 7-12：
| task | 状态 |
|---|---|
| 7 数据生成（normal 42-73 / induced 42-47，38 runs） | ✅ |
| 8 phase_baseline 模块 + 测试 | ✅（61 passed） |
| 9 冻结规则 A/B/C（knots + MAD floor + McNemar + 两 q + condition 筛） | ✅ |
| 10 held-out 真跑 + 分析（validate_inputs 闸门 + Wilson + McNemar） | ✅ |
| 11 v0.7.5 benchmark 文档 | ✅ |
| 12 memory + diff（不 push） | 🔄 进行中 |

## 7. 关键文件（v0.7.5 新增/修改）
- `soap/apps/training/phase_baseline.py`（phase-aware A/B/C + knots + McNemar + validate_inputs 闸门）
- `soap/apps/training/hf_repr_experiment.py`（v0.7.4 + local_files_only=True）
- `soap/apps/training/hf_collapse_calibration.py`（v0.7.4，local_files_only 一致）
- `docs/v0.7.5_phase_aware_benchmark.md` / `docs/v0.7.5_experiment_spec.md` / `docs/v0.7.5_calibration_report.md`
- `tests/test_phase_baseline.py`（8 用例）
- `examples/hf_repr_seed{42-73}_normal.csv`（32）/ `hf_repr_seed{42-47}_condition_b.csv`（6）

## 8. 下一步
- task12：audit + 本地 commit（不 push）
- v0.7.6（候选）：class-conditional geometry（延后自 v0.7.5）
- v0.8：统一 CLI / report（封装两层诊断；phase-aware B 替代 static A 进默认路径）

## 9. 管家模式产出（v0.7.5）
- MiMo 写 phase_baseline.py（重写 + 多轮审核）+ test_phase_baseline（重写 + 恢复断言）+ v0.7.5 文档；主线程审核 + 真跑 + 机制/接口修复（condition 筛 7 处 / load_runs glob / mcnemar n01/n10 / validate_inputs 闸门 / local_files_only）。
- mimo_worker `trust_env=False`（v0.7.4 沉淀）解决 macOS 系统代理 ProxyError。

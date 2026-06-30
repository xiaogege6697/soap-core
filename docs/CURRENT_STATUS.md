# SOAP-Core Current Status（v0.7.6 完成版）

- 日期：2026-06-30
- 基线：v0.7.5 `b3abde2`（已 push origin/main）+ v0.7.6 本地 commit（未 push）
- v0.7.6 class-conditional geometry 完成

## 1. v0.7.6 task 状态（全完成）
| task | 状态 |
|---|---|
| 13 hf_classcond_experiment.py + 38 runs | ✅ 修 9 bug + smoke 7 断言 + 38 runs 串行完成 |
| 14 classcond_metrics + 单测 | ✅ 66 passed |
| 15 分析 induced vs normal | ✅ 结论见 §2 |
| 16 文档 | ✅ docs/v0.7.6_classcond_geometry_benchmark.md |
| 17 audit + commit | ✅ pytest 66 / 敏感扫描净 / 数据 38 完整 / 本地 commit（不 push）|

## 2. v0.7.6 核心结论（详见 benchmark 文档）
induced（controlled rank-1）在可训练层（尤 layer_4）**几何坍缩**：NC1↓ + centroid eff rank↓（layer_4 →1.1，rank-1 压维直接证据）+ ETF↑；但 **val_acc 持平 1.0**（normal 前向；仅说明当前合成五分类任务可完成，不可推断模型通用功能/自然鲁棒性/权重功能保持）+ **Fisher 暴涨**（rank-1 压 within-class 的数学假象，不可解读为分离改善）。

→ induced 是 controlled rank-1 **注入的几何 collapse**；当前合成五分类 val_acc 仍 1.0（一维表示即可完成该简单任务，不可推断权重/通用功能保持）；**非典型良性 NC**（无 ETF 收敛/rank 保持）**也非典型病态**（accuracy 未恶化）。**结论：class-conditional geometry 在当前 controlled benchmark 中仍不能判定良性/病态自然 collapse**（rank-1 与 geometry 数学对齐）。承接 v0.7.5 边界。

## 3. v0.7.6 文件清单
- `soap/apps/training/classcond_metrics.py`（NC1/centroid rank/Fisher/ETF 公式，冻结）
- `soap/apps/training/classcond_analysis.py`（末段聚合 + 方向判定）
- `soap/apps/training/hf_classcond_experiment.py`（实验脚本，修 9 bug：tuple 解包/num_classes/probe 泄漏/tokenize 参数序/accuracy 独立列/...）
- `tests/test_classcond_metrics.py`（5 用例）+ `tests/smoke_classcond.py`（端到端 7 断言）
- `docs/v0.7.6_experiment_spec.md`（公式冻结 spec）+ `docs/v0.7.6_classcond_geometry_benchmark.md`（科学文档）
- `examples/cc_seed*_*.csv`（38，每 run 280 数据行 = 40ckpt×7层）+ `examples/run_all_classcond.sh`（串行/并行调度）+ `examples/_classcond_analysis.txt`（分析输出）

## 4. 教训（2026-06-30，已入 memory）
- **py_compile + 单测 ≠ 主入口真跑**：脚本 9 bug 未被"审核"发现，`tests/smoke_classcond.py` 端到端 7 断言兜底。
- **Mac 8GB 内存**：批量 run 必须**串行**（并行 5 → swap 12.95GB 烧 SSD + 卡顿）。
- **Mac 睡眠** → multiprocessing semaphore 泄漏 → python 卡 shutdown、CSV 不完整；`caffeinate -i` 防睡眠。

## 5. 下一步
- **v0.8 候选**：统一 CLI，phase-aware B 进默认；v0.7.6 语义证据已得（class-conditional **不进默认**，仅 controlled benchmark 内）。
- **不外推**：跨任务 / 跨模型 / 自然 mode collapse 均未验证；checkpoint 非独立不宣称 conformal。

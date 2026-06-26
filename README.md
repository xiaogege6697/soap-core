# SOAP-Core

**Scale-Optimized Attractor Prediction Core**

SOAP-Core 是一个领域无关的复杂系统分析底座，用于从高维时间序列中：

1. 搜索最优有效维度 `d*`
2. 重构低维状态空间
3. 识别吸引子候选
4. 估计状态转移概率
5. 输出概率未来与分析报告

它不绑定中医、金融、气候、社会或 AI 训练系统；这些都应作为应用层接入。

## MVP 输入

CSV 时间序列：

```csv
timestamp,feature_1,feature_2,feature_3
2026-01-01,0.1,1.2,3.4
2026-01-02,0.2,1.0,3.8
```

第一列可为时间戳，也可以没有时间戳。其余列必须为数值。

## MVP 输出

```text
outputs/
  dimension_scores.csv
  dimension_scores.svg
  embedding.csv
  embedding.svg
  transition_matrix.csv
  attractor_states.csv
reports/
  report.md
```

## 快速运行

安装科学计算依赖：

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[science]"
```

生成示例数据：

```bash
python3 -m soap.cli generate-example --output examples/synthetic_cyclic.csv
```

生成 Lorenz 混沌示例数据：

```bash
python3 -m soap.cli generate-lorenz --output examples/synthetic_lorenz.csv
```

分析示例数据：

```bash
python3 -m soap.cli analyze examples/synthetic_cyclic.csv --max-dim 8 --clusters 3
```

分析 Lorenz 数据：

```bash
python3 -m soap.cli analyze examples/synthetic_lorenz.csv --max-dim 8 --clusters 3 --output-dir outputs_lorenz --report-dir reports_lorenz
```

使用 PCA + 时间顺序 train/test 预测评分：

```bash
.venv/bin/python -m soap.cli analyze examples/synthetic_lorenz.csv --max-dim 8 --clusters 3 --method pca --output-dir outputs_lorenz --report-dir reports_lorenz
```

使用 Takens delay embedding + AMI/FNN 自动参数估计：

```bash
.venv/bin/python -m soap.cli analyze examples/synthetic_lorenz.csv --embedding takens --delay auto --embedding-dim auto --max-delay 50 --max-embedding-dim 8 --method pca --clusters 3 --output-dir outputs_v04_smoke --report-dir reports_v04_smoke
```

v0.5 已接入 Simplex / S-Map 预测诊断与 recurrence 诊断，默认 `--predictor none` 不加载预测模块：

```bash
.venv/bin/python -m soap.cli analyze examples/synthetic_lorenz.csv --embedding takens --method pca --predictor none --output-dir outputs_v05_compat --report-dir reports_v05_compat
```

启用 Simplex / S-Map 与 recurrence diagnostics（对应模块均已存在）：

```bash
.venv/bin/python -m soap.cli analyze examples/synthetic_lorenz.csv --embedding takens --method pca --predictor simplex --neighbors 12 --recurrence --recurrence-quantile 0.1 --output-dir outputs_v05_diag --report-dir reports_v05_diag
```

当前 CLI 会优先接入 `soap.prediction.simplex`、`soap.prediction.smap` 与 `soap.metrics.recurrence`；若对应模块或 API 尚不存在，会跳过对应诊断并在报告中写明 `unavailable`，不会影响默认 `--predictor none` / 不启用 `--recurrence` 的旧命令。

## 核心原则

SOAP-Core 预测的不是未来精确点位，而是：

- 吸引子归属
- 状态转移概率
- 临界风险
- 低维演化趋势
- 干预后可能路径

## 当前版本边界

当前 baseline 使用纯 Python 标准库实现，便于零依赖启动。v0.3 已加入可选科学计算路径，v0.4 已加入相空间重构路径。当前核心使用：

- `numpy`
- `scikit-learn`
- Takens delay embedding
- AMI delay estimation
- FNN embedding dimension estimation
- CLI/report 级 Prediction Diagnostics 与 Recurrence Diagnostics 已集成（Simplex、S-Map、Recurrence 模块均已接入）

`science` extra 已预装后续数据层和可视化所需依赖：

- `pandas`
- `matplotlib`

后续版本会继续加入：

- `umap-learn`
- `giotto-tda` / `ripser`
- `streamlit`

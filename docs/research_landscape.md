# SOAP-Core Research Landscape

调研日期：2026-06-25

## 1. 结论

尚未发现成熟项目直接占住“领域无关的复杂系统状态空间 / 吸引子 / 临界转变 / 概率未来预测”定位。

现有生态分为两类：

1. 学术/开源方法库：强算法、弱产品统一性。
2. 行业垂直产品：强场景、弱通用状态空间解释。

SOAP-Core 的机会在中间层：把动力系统、EDM、TDA、intrinsic dimension、early warning signals 等方法产品化成通用引擎。

## 2. 方法谱系

```text
Intrinsic Dimension
  ↓
Attractor Reconstruction
  ↓
Attractor Diagnostics
  ↓
Prediction Head
  ↓
Critical Transition
  ↓
Product API
```

## 3. 优先学习项目

| 优先级 | 项目 | 价值 |
|---:|---|---|
| 1 | pyEDM | 最接近“相空间重构 → 非线性预测/因果” |
| 2 | DynamicalSystems.jl | 学习统一动力系统软件架构 |
| 3 | DelayEmbeddings.jl | 学习 delay embedding 参数选择 |
| 4 | DADApy | 补齐 high-dimensional intrinsic dimension |
| 5 | pyunicorn | recurrence network / RQA / 复杂系统网络化 |

## 4. 相邻项目

- pyEDM / rEDM：EDM、Simplex、S-Map、CCM、Multiview。
- DynamicalSystems.jl：非线性动力系统生态。
- DelayEmbeddings.jl：相空间重构。
- giotto-tda：Takens embedding + persistent homology。
- teaspoon：TDA + nonlinear time series。
- NoLiTSA / nolds：Lyapunov、FNN、correlation dimension 等指标。
- scikit-dimension / DADApy：intrinsic dimension。
- pyunicorn：recurrence network、climate networks。
- ewstools：early warning signals。

## 5. 产品差异化

SOAP-Core 不做又一个指标 dashboard。

目标输出：

1. 当前状态坐标
2. 稳定 basin / attractor 归属
3. 临界边界距离
4. 未来风险路径 / transition probability

## 6. 首个应用场景建议

首选：AI 训练不稳定预警。

原因：

- 数据天然存在：loss、grad norm、learning rate、activation stats、GPU telemetry。
- 决策闭环短：暂停、降学习率、回滚 checkpoint。
- 竞品 W&B / TensorBoard 偏 dashboard 和阈值告警，较少状态空间解释。
- 可用历史 training run 回放验证是否更早识别 divergence / overfit / mode collapse。

备选：工业 predictive maintenance。

## 7. 路线图建议

### v0.3 科学计算基础

- pandas/numpy 数据层
- sklearn PCA
- train/test predictive scoring
- 更严谨的 score curve

### v0.4 相空间重构

- Takens delay embedding
- AMI 估计 delay
- FNN 估计 embedding dimension

### v0.5 吸引子诊断

- recurrence plot
- Lyapunov proxy
- basin transition matrix
- early warning indicators

### v0.6 应用验证

- AI training run dataset adapter
- divergence / instability detection benchmark

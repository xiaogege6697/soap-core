"""Tests for soap.apps.training.phase_baseline v0.7.5 frozen interface."""

import math
import numpy as np
import pytest
from soap.apps.training.phase_baseline import (
    load_runs,
    phase_baseline,
    interpolate_baseline,
    anomaly_score,
    compute_run_scores_phase,
    compute_static_scores_per_layer,
    run_score_static,
    detect_static,
    mcnemar_exact,
    wilson_ci,
    np_percentile,
)


def make_run(seed, condition, eff_rank_traj_by_layer):
    """eff_rank_traj_by_layer: {layer_name: [40 个 eff_rank 值]}。返回 {"seed","condition","rows"}。"""
    rows = []
    steps = [f"step_{i}" for i in range(3, 121, 3)]  # 40 checkpoints
    for layer, traj in eff_rank_traj_by_layer.items():
        for i, er in enumerate(traj):
            er = max(er, 1e-10)
            rows.append({
                "timestamp": steps[i],
                "layer_name": layer,
                "effective_rank": float(er),
                "collapse_score": 1.0 / er,
            })
    return {"seed": seed, "condition": condition, "rows": rows}


def test_wilson_ci():
    """Wilson 置信区间：边界与退化情况"""
    # k=0, n=8 → 下界接近 0
    lo, hi = wilson_ci(0, 8)
    assert lo == pytest.approx(0.0, abs=0.05)

    # k=8, n=8 → 上界接近 1
    lo, hi = wilson_ci(8, 8)
    assert hi == pytest.approx(1.0, abs=0.05)

    # k=4, n=8 → CI 包含 0.5
    lo, hi = wilson_ci(4, 8)
    assert lo <= 0.5 <= hi

    # n=0 → 返回 (0, 1)
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0
    assert hi == 1.0


def test_mcnemar_exact():
    """McNemar 精确检验：不一致对计数与 p 值"""
    da = [1, 0, 1, 1, 0]
    db = [0, 0, 1, 0, 0]
    result = mcnemar_exact(da, db)
    # 不一致对：i=0 (1,0) 和 i=3 (1,0) → n10=2, n01=0, n_discordant=2
    assert result["n01"] == 0
    assert result["n10"] == 2
    assert result["n_discordant"] == 2
    assert 0.0 <= result["p_value"] <= 1.0

    # 完全一致 → n_discordant=0, p_value=1.0
    same = [1, 0, 1, 0, 1]
    result2 = mcnemar_exact(same, same)
    assert result2["n_discordant"] == 0
    assert result2["p_value"] == pytest.approx(1.0)


def test_phase_baseline_structure():
    """phase_baseline 返回 knots / layer_floor 结构校验"""
    rng = np.random.RandomState(42)

    # 2 个 dev normal run，每 run 含 layer_0 / layer_1
    run1 = make_run(seed=1, condition="normal", eff_rank_traj_by_layer={
        "layer_0": list(np.abs(rng.randn(40)) + 3.0),
        "layer_1": list(np.abs(rng.randn(40)) + 5.0),
    })
    run2 = make_run(seed=2, condition="normal", eff_rank_traj_by_layer={
        "layer_0": list(np.abs(rng.randn(40)) + 3.0),
        "layer_1": list(np.abs(rng.randn(40)) + 5.0),
    })

    baseline = phase_baseline([run1, run2], dev_normal_seeds=[1, 2], n_knots=4)

    # 顶层字段
    assert baseline["n_knots"] == 4
    assert len(baseline["knot_positions"]) == 4
    assert len(baseline["knots"]) == 4

    # 每个 knot 含 position / median / mad
    for knot in baseline["knots"]:
        assert {"position", "median", "mad"} <= set(knot.keys())
        for layer in ("layer_0", "layer_1"):
            assert layer in knot["median"]
            assert layer in knot["mad"]

    # layer_floor 含所有 layer
    for layer in ("layer_0", "layer_1"):
        assert layer in baseline["layer_floor"]


def test_anomaly_score():
    """log(er) == median 时 anomaly_score 接近 0"""
    median_val = math.log(5.0)  # median 在 log 空间
    mad_val = 0.5

    baseline = {
        "n_knots": 2,
        "knot_positions": [0.0, 1.0],
        "knots": [
            {"position": 0.0, "median": {"layer_0": median_val}, "mad": {"layer_0": mad_val}},
            {"position": 1.0, "median": {"layer_0": median_val}, "mad": {"layer_0": mad_val}},
        ],
        "layer_floor": {"layer_0": 0.01},
    }

    # phase_t=0.5 插值 → median=median_val, mad=mad_val
    # eff_rank=5.0 → log(5.0)==median_val → score≈0
    score = anomaly_score(5.0, "layer_0", 0.5, baseline)
    assert abs(score) < 0.01


def test_compute_static_scores_per_layer_condition():
    """condition='normal' 只取 normal run，condition_b 不混入"""
    rng = np.random.RandomState(0)

    # 2 个 normal run，eff_rank 集中在 ~5.0
    run_n1 = make_run(seed=1, condition="normal", eff_rank_traj_by_layer={
        "layer_0": list(np.full(40, 5.0) + rng.randn(40) * 0.1),
    })
    run_n2 = make_run(seed=2, condition="normal", eff_rank_traj_by_layer={
        "layer_0": list(np.full(40, 5.0) + rng.randn(40) * 0.1),
    })

    # 同 seed=1 但 condition_b，eff_rank=1000（极端不同）
    run_b = make_run(seed=1, condition="condition_b", eff_rank_traj_by_layer={
        "layer_0": list(np.full(40, 1000.0)),
    })

    runs_all = [run_n1, run_n2, run_b]
    runs_pure = [run_n1, run_n2]

    # 含 condition_b 的列表 vs 纯 normal 列表，均指定 condition="normal"
    thr_all = compute_static_scores_per_layer(runs_all, seeds=[1, 2], condition="normal")
    thr_pure = compute_static_scores_per_layer(runs_pure, seeds=[1, 2], condition="normal")

    # condition_b 不应污染结果
    assert thr_all["layer_0"] == pytest.approx(thr_pure["layer_0"])
    # 结果基于 ~5.0 而非 ~1000
    assert thr_all["layer_0"] < 100.0


def test_np_percentile():
    """np_percentile 边界与线性插值（恢复重写时误删的独立断言）"""
    assert np_percentile([1, 2, 3, 4], 0) == 1
    assert np_percentile([1, 2, 3, 4], 100) == 4
    assert np_percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert np_percentile([], 50) == 0.0


def test_detect_static():
    """detect_static（method_A 单 run）：超阈值检出，未超不检出"""
    run_hit = {"rows": [
        {"layer_name": "layer_0", "collapse_score": 0.1},
        {"layer_name": "layer_1", "collapse_score": 0.9},
    ]}
    run_miss = {"rows": [{"layer_name": "layer_0", "collapse_score": 0.1}]}
    thresholds = {"layer_0": 0.5, "layer_1": 0.5}
    assert detect_static(run_hit, thresholds) == 1   # layer_1=0.9 >= 0.5
    assert detect_static(run_miss, thresholds) == 0  # 仅 layer_0=0.1 未超


def test_compute_run_scores_phase():
    """compute_run_scores_phase：run_score=max anomaly + condition 筛"""
    baseline = {
        "n_knots": 2, "knot_positions": [0.0, 1.0],
        "knots": [
            {"position": 0.0, "median": {"layer_0": 0.0}, "mad": {"layer_0": 1.0}},
            {"position": 1.0, "median": {"layer_0": 0.0}, "mad": {"layer_0": 1.0}},
        ],
        "layer_floor": {"layer_0": 0.1},
    }
    # eff_rank=e^5 → log=5，median=0，mad=1 → anomaly≈5
    run = make_run(seed=1, condition="normal",
                   eff_rank_traj_by_layer={"layer_0": [math.exp(5)] * 40})
    scores = compute_run_scores_phase([run], [1], baseline, "normal")
    assert 1 in scores
    run_score, _ = scores[1]
    assert run_score == pytest.approx(5.0, abs=0.2)
    # condition 筛：condition_b 不匹配 normal
    assert 1 not in compute_run_scores_phase([run], [1], baseline, "condition_b")

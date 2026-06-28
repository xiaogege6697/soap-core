"""Phase-aware baseline deviation detection for SOAP v0.7.5 frozen release.

Phase-knot baseline (8 uniform knots), MAD layer floor, exact McNemar paired comparison,
run-level calibrated false-alarm suppression (methods A/B/C). Thresholds calibrated on
dev normal (42-53) only; held-out sets (54-73 normal, 45-47 induced) for final evaluation only.
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict
from typing import Any

# ==============================================================================
# 工具函数
# ==============================================================================

def load_runs(data_dir):
    """加载 examples/hf_repr_seed*_*.csv，返回 [{"seed","condition","rows"}]。
    文件名格式 hf_repr_seed{N}_{condition}.csv；数值列转 float/int。
    """
    runs = []
    pattern = os.path.join(data_dir, "hf_repr_seed*_*.csv")
    for fp in sorted(glob.glob(pattern)):
        fn = os.path.basename(fp)
        m = re.match(r"hf_repr_seed(\d+)_(.+)\.csv", fn)
        if not m:
            continue
        seed = int(m.group(1))
        condition = m.group(2)
        rows = []
        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["seed"] = int(row["seed"])
                row["timestamp"] = row["timestamp"].strip()
                row["layer_name"] = row["layer_name"].strip()
                for c in ("loss", "val_loss", "grad_norm", "learning_rate",
                          "effective_rank", "representation_variance", "collapse_score"):
                    row[c] = float(row[c])
                rows.append(row)
        rows.sort(key=lambda r: r["timestamp"])
        runs.append({"seed": seed, "condition": condition, "rows": rows})
    return runs


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def np_percentile(arr, q):
    """NumPy-compatible percentile (linear interpolation, no dependency)."""
    if len(arr) == 0:
        return 0.0
    s = sorted(arr)
    k = (len(s) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def parse_seed_range(seed_range_str):
    """解析 '42-53' 或 '42,43,44' 为 seed 列表。"""
    seeds = []
    for part in seed_range_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def format_seed_list(seeds):
    """格式化 seed 列表为紧凑字符串。"""
    if not seeds:
        return ""
    seeds_s = sorted(seeds)
    ranges = []
    start = seeds_s[0]
    end = start
    for s in seeds_s[1:]:
        if s == end + 1:
            end = s
        else:
            ranges.append((start, end))
            start = end = s
    ranges.append((start, end))
    return ",".join(f"{s}-{e}" if s != e else f"{s}" for s, e in ranges)


# ==============================================================================
# 评分辅助（v0.7.4 静态方法）
# ==============================================================================

def compute_static_scores_per_layer(runs, seeds, condition="normal"):
    """计算每层 collapse_score 的指定分位数（仅 condition 匹配的 run）。"""
    layer_vals = defaultdict(list)
    for run in runs:
        if run["seed"] not in seeds or run["condition"] != condition:
            continue
        for row in run["rows"]:
            try:
                v = float(row["collapse_score"])
                layer_vals[row["layer_name"]].append(v)
            except (ValueError, KeyError):
                continue
    return {layer: np_percentile(vals, 99) for layer, vals in layer_vals.items()}


def run_score_static(run, thresholds):
    """静态方法 run_score = max over all (layer, ckpt) of collapse_score."""
    max_score = 0.0
    for row in run["rows"]:
        try:
            v = float(row["collapse_score"])
        except (ValueError, KeyError):
            continue
        ln = row["layer_name"]
        if v > max_score:
            max_score = v
    return max_score


def detect_static(run, thresholds):
    """静态方法：任意 (layer, ckpt) 超阈值即检出。"""
    for row in run["rows"]:
        ln = row["layer_name"]
        try:
            v = float(row["collapse_score"])
        except (ValueError, KeyError):
            continue
        if ln in thresholds and v >= thresholds[ln]:
            return 1
    return 0


# ==============================================================================
# Phase-aware 工具函数
# ==============================================================================

def parse_phase_t(run_rows):
    """从 timestamp 解析 phase_t（checkpoint 序号/总 checkpoint 数）。
    
    按 timestamp 排序后，第 i 个唯一 timestamp 的 phase_t = i / (n-1)。
    每行的 phase_t 取其 timestamp 对应的排序索引归一化值。
    """
    timestamps = sorted(set(row["timestamp"] for row in run_rows))
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    n_ckpt = len(timestamps)
    for row in run_rows:
        idx = ts_to_idx[row["timestamp"]]
        row["_phase_t"] = idx / max(n_ckpt - 1, 1)
    return run_rows


def layer_names_from_baseline(baseline):
    """返回基线中所有 layer 名称（固定顺序）。"""
    if not baseline["knots"]:
        return []
    return sorted(baseline["knots"][0]["median"].keys())


# ==============================================================================
# Phase baseline 核心
# ==============================================================================

def phase_baseline(runs, dev_normal_seeds, n_knots=8):
    """计算 phase-aware 基线：8 均匀 knots 上的 median 与 MAD_scaled。
    
    knots 均匀分布于 t=0 到 t=1（共 n_knots 个，间距 1/(n_knots-1)）。
    每个 (layer, knot) 计算 dev normal 的 log(effective_rank) 的 median 和
    MAD_scaled = 1.4826 * MAD。
    
    MAD layer floor：每层插值后 MAD 取 max(interpolated, floor_layer)，
    floor_layer = 该层 dev normal 全体 log(eff_rank) 的 raw MAD_scaled。
    """
    # 确定 knots 位置
    knot_positions = [i / max(n_knots - 1, 1) for i in range(n_knots)]

    # 收集每个 (layer, knot_index) 的 log(eff_rank) 值
    layer_knot_vals = defaultdict(lambda: defaultdict(list))

    for run in runs:
        if run["seed"] not in dev_normal_seeds or run["condition"] != "normal":
            continue
        run_rows = parse_phase_t(run["rows"])
        for row in run_rows:
            try:
                eff_rank = float(row["effective_rank"])
                if eff_rank <= 0:
                    continue
                log_er = math.log(eff_rank)
            except (ValueError, KeyError):
                continue
            pt = row["_phase_t"]
            ln = row["layer_name"]

            # 确定最近 knot
            best_k = 0
            best_d = abs(pt - knot_positions[0])
            for k in range(1, n_knots):
                d = abs(pt - knot_positions[k])
                if d < best_d:
                    best_d = d
                    best_k = k
            layer_knot_vals[ln][best_k].append(log_er)

    # 计算每层的 raw MAD_scaled（floor 基础）
    layer_raw_mad = {}
    layer_all_vals = defaultdict(list)
    for ln, knot_data in layer_knot_vals.items():
        for k, vals in knot_data.items():
            layer_all_vals[ln].extend(vals)
    for ln, vals in layer_all_vals.items():
        if len(vals) >= 2:
            med = sorted(vals)[len(vals) // 2]
            abs_devs = [abs(v - med) for v in vals]
            layer_raw_mad[ln] = 1.4826 * sorted(abs_devs)[len(abs_devs) // 2]
        else:
            layer_raw_mad[ln] = 0.0

    # 计算每个 knot 的 median 和 MAD_scaled
    knots_data = []
    for k in range(n_knots):
        knot_med = {}
        knot_mad = {}
        all_layers = sorted(set(ln for ln in layer_knot_vals.keys()))
        for ln in all_layers:
            vals = layer_knot_vals[ln].get(k, [])
            if len(vals) >= 2:
                s = sorted(vals)
                med = s[len(s) // 2]
                abs_devs = [abs(v - med) for v in vals]
                mad = 1.4826 * sorted(abs_devs)[len(abs_devs) // 2]
            elif len(vals) == 1:
                med = vals[0]
                mad = 0.0
            else:
                med = 0.0
                mad = 0.0
            knot_med[ln] = med
            knot_mad[ln] = mad
        knots_data.append({
            "position": knot_positions[k],
            "median": knot_med,
            "mad": knot_mad,
        })

    return {
        "n_knots": n_knots,
        "knot_positions": knot_positions,
        "knots": knots_data,
        "layer_floor": layer_raw_mad,
    }


def interpolate_baseline(baseline, phase_t):
    """线性插值基线 median 和 MAD_floored 到任意 phase_t。
    
    t<0 用 knot0，t>1 用最后一个 knot。
    MAD floor：每层取 max(interpolated, floor_layer)。
    """
    knots = baseline["knots"]
    n = len(knots)
    floor = baseline["layer_floor"]
    t = max(0.0, min(1.0, phase_t))

    # 选插值区间
    if n < 2:
        k = knots[0]
        return {"median": dict(k["median"]), "mad": {ln: max(k["mad"].get(ln, 0.0), floor.get(ln, 0.0)) for ln in k["median"]}}

    ki = 0
    for i in range(n - 1):
        if t >= knots[i]["position"] and t <= knots[i + 1]["position"]:
            ki = i
            break
    else:
        ki = n - 2 if t > knots[-1]["position"] else 0

    k0 = knots[ki]
    k1 = knots[ki + 1]
    span = k1["position"] - k0["position"]
    if span <= 0:
        alpha = 0.0
    else:
        alpha = (t - k0["position"]) / span

    median_interp = {}
    mad_floored = {}
    all_layers = sorted(set(list(k0["median"].keys()) + list(k1["median"].keys())))
    for ln in all_layers:
        m0 = k0["median"].get(ln, 0.0)
        m1 = k1["median"].get(ln, 0.0)
        median_interp[ln] = m0 + alpha * (m1 - m0)

        a0 = k0["mad"].get(ln, 0.0)
        a1 = k1["mad"].get(ln, 0.0)
        mad_interp = a0 + alpha * (a1 - a0)
        mad_floored[ln] = max(mad_interp, floor.get(ln, 0.0))

    return {"median": median_interp, "mad": mad_floored}


def anomaly_score(eff_rank, layer, phase_t, baseline):
    """Phase-aware anomaly score: |log(eff_rank) - median_interp| / (mad_floored_interp + eps)。"""
    eps = 1e-8
    log_er = math.log(max(eff_rank, 1e-12))
    interp = interpolate_baseline(baseline, phase_t)
    med = interp["median"].get(layer, 0.0)
    mad = interp["mad"].get(layer, 0.0)
    return abs(log_er - med) / (mad + eps)


# ==============================================================================
# 运行评分
# ==============================================================================

def compute_run_scores_phase(runs, seeds, baseline, condition):
    """为指定 seeds 且 condition 匹配的 run 计算 phase 方法的运行评分。

    phase_t 始终从 timestamp 解析（不用行索引 i/279）。
    返回 dict[seed] -> (run_score, layer_ckpt_scores)。
    """
    results = {}
    for run in runs:
        if run["seed"] not in seeds or run["condition"] != condition:
            continue
        run_rows = parse_phase_t(run["rows"])
        max_score = 0.0
        layer_ckpt = defaultdict(list)
        for row in run_rows:
            try:
                er = float(row["effective_rank"])
            except (ValueError, KeyError):
                continue
            if er <= 0:
                continue
            ln = row["layer_name"]
            pt = row["_phase_t"]
            score = anomaly_score(er, ln, pt, baseline)
            layer_ckpt[ln].append({"phase_t": pt, "score": score})
            if score > max_score:
                max_score = score
        results[run["seed"]] = (max_score, dict(layer_ckpt))
    return results


# ==============================================================================
# 三方法检测
# ==============================================================================

def detect_method_A(run, thresholds):
    """方法 A：v0.7.4 静态逐层阈值 + run-level any_layer。"""
    return detect_static(run, thresholds)


def detect_method_B(run_score, threshold):
    """方法 B：phase-knots 基线 + run_score 超 run_q 阈值即检出。"""
    return 1 if run_score >= threshold else 0


def detect_method_C(run_data, run_threshold, anomaly_threshold, min_persist=3):
    """方法 C：B + persistent（某 layer 连续 >=min_persist checkpoint 超阈值）。"""
    run_score, layer_ckpt = run_data
    if run_score < run_threshold:
        return 0
    for ln, ckpts in layer_ckpt.items():
        ckpts_sorted = sorted(ckpts, key=lambda c: c["phase_t"])
        streak = 0
        for c in ckpts_sorted:
            if c["score"] >= run_threshold:
                streak += 1
                if streak >= min_persist:
                    return 1
            else:
                streak = 0
    return 0


# ==============================================================================
# Exact McNemar
# ==============================================================================

def mcnemar_exact(detect_a, detect_b):
    """Exact McNemar paired comparison (binomial exact, 双侧)。
    
    n01: a=0,b=1 的数量; n10: a=1,b=0 的数量。
    n_discordant = n01 + n10。
    p = 2 * sum(C(n,k) * 0.5^n) for k <= min(n01, n10)，双侧精确。
    """
    pairs = list(zip(detect_a, detect_b))
    n01 = sum(1 for a, b in pairs if a == 0 and b == 1)
    n10 = sum(1 for a, b in pairs if a == 1 and b == 0)
    n_discordant = n01 + n10

    if n_discordant == 0:
        p_value = 1.0
    else:
        k_min = min(n01, n10)
        p_value = 0.0
        for k in range(k_min + 1):
            p_value += math.comb(n_discordant, k) * (0.5 ** n_discordant)
        p_value = min(1.0, 2.0 * p_value)

    return {"n01": n01, "n10": n10, "n_discordant": n_discordant, "p_value": p_value}


# ==============================================================================
# 输入完整性闸门（task10 前必过，任一条件不满足 raise，不静默跳过）
# ==============================================================================

def validate_inputs(runs, dev_normal_seeds, held_normal_seeds,
                    dev_induced_seeds, held_induced_seeds):
    """严格校验输入完整性：run 数、行数、(seed,condition) 唯一、dev/held 不交叉、
    NaN/Inf=0、checkpoint/layer 集合一致。任一失败 raise AssertionError。
    """
    expected_counts = [
        ("dev",  "normal",   sorted(set(dev_normal_seeds)),   12, "normal"),
        ("held", "normal",   sorted(set(held_normal_seeds)),  20, "normal"),
        ("dev",  "induced",  sorted(set(dev_induced_seeds)),   3, "condition_b"),
        ("held", "induced",  sorted(set(held_induced_seeds)),  3, "condition_b"),
    ]

    # 1) (seed, condition) 无重复
    seen = set()
    for r in runs:
        key = (r["seed"], r["condition"])
        if key in seen:
            raise AssertionError(f"重复 run (seed,condition)={(r['seed'], r['condition'])}")
        seen.add(key)

    # 2) 每 run 280 rows + NaN/Inf=0 + checkpoint/layer 集合一致
    ref_ckpts = ref_layers = None
    for r in runs:
        rows = r["rows"]
        if len(rows) != 280:
            raise AssertionError(
                f"(seed={r['seed']},{r['condition']}) rows={len(rows)} != 280 (40ckpt×7layer)")
        cur_ckpts = sorted(set(row["timestamp"] for row in rows))
        cur_layers = sorted(set(row["layer_name"] for row in rows))
        if ref_ckpts is None:
            ref_ckpts, ref_layers = cur_ckpts, cur_layers
        else:
            if cur_ckpts != ref_ckpts:
                raise AssertionError(f"(seed={r['seed']},{r['condition']}) checkpoint 集合不一致")
            if cur_layers != ref_layers:
                raise AssertionError(f"(seed={r['seed']},{r['condition']}) layer 集合不一致")
        for row in rows:
            for c in ("effective_rank", "representation_variance", "collapse_score",
                      "loss", "val_loss", "grad_norm"):
                v = row.get(c, 0.0)
                if not math.isfinite(float(v)):
                    raise AssertionError(
                        f"(seed={r['seed']},{r['condition']}) {c}={v} 非有限(NaN/Inf)")

    # 3) 各分组 run 数严格匹配
    for split, kind, seeds, n_expected, cond in expected_counts:
        matched = [r for r in runs if r["seed"] in set(seeds) and r["condition"] == cond]
        if len(matched) != n_expected:
            raise AssertionError(
                f"{split} {kind} run 数={len(matched)} != {n_expected}")

    # 4) dev/held seed 无交叉（normal 与 induced 各自）
    if set(dev_normal_seeds) & set(held_normal_seeds):
        raise AssertionError(f"dev/held normal seed 交叉: {set(dev_normal_seeds) & set(held_normal_seeds)}")
    if set(dev_induced_seeds) & set(held_induced_seeds):
        raise AssertionError(f"dev/held induced seed 交叉: {set(dev_induced_seeds) & set(held_induced_seeds)}")

    return True


# ==============================================================================
# 评估
# ==============================================================================

def evaluate(runs, dev_normal_seeds, held_normal_seeds, held_induced_seeds, baseline,
             thresholds_A, run_threshold_B, run_threshold_C, run_q):
    """对 held-out 数据进行评估。
    
    输出 A/B/C 误报数+Wilson CI+检出数+McNemar 精确配对比较。
    """
    held_normal_runs = [r for r in runs if r["seed"] in held_normal_seeds and r["condition"] == "normal"]
    held_induced_runs = [r for r in runs if r["seed"] in held_induced_seeds and r["condition"] == "condition_b"]

    # 方法 A
    det_A = [detect_method_A(r, thresholds_A) for r in held_normal_runs]
    ind_det_A = [detect_method_A(r, thresholds_A) for r in held_induced_runs]

    # 方法 B
    scores_normal = compute_run_scores_phase(runs, held_normal_seeds, baseline, "normal")
    scores_induced = compute_run_scores_phase(runs, held_induced_seeds, baseline, "condition_b")
    det_B = [detect_method_B(scores_normal.get(s, (0.0, {}))[0], run_threshold_B)
             for s in held_normal_seeds]
    ind_det_B = [detect_method_B(scores_induced.get(s, (0.0, {}))[0], run_threshold_B)
                 for s in held_induced_seeds]

    # 方法 C
    det_C = [detect_method_C(scores_normal.get(s, (0.0, {})), run_threshold_C, run_threshold_C)
             for s in held_normal_seeds]
    ind_det_C = [detect_method_C(scores_induced.get(s, (0.0, {})), run_threshold_C, run_threshold_C)
                 for s in held_induced_seeds]

    n = len(held_normal_runs)

    # McNemar 两两配对
    mcnemar_results = {}
    for label, da, db in [("A_vs_B", det_A, det_B), ("A_vs_C", det_A, det_C), ("B_vs_C", det_B, det_C)]:
        mcnemar_results[label] = mcnemar_exact(da, db)

    return {
        "run_q": run_q,
        "n_normal": n,
        "n_induced": len(held_induced_runs),
        "fa_A": sum(det_A), "det_A": sum(ind_det_A),
        "fa_B": sum(det_B), "det_B": sum(ind_det_B),
        "fa_C": sum(det_C), "det_C": sum(ind_det_C),
        "ci_A": wilson_ci(sum(det_A), n),
        "ci_B": wilson_ci(sum(det_B), n),
        "ci_C": wilson_ci(sum(det_C), n),
        "mcnemar": mcnemar_results,
        "det_A_vec": det_A, "det_B_vec": det_B, "det_C_vec": det_C,
    }


# ==============================================================================
# 报告
# ==============================================================================

def write_markdown_report(output_path, primary, sensitivity, dev_normal_seeds, held_normal_seeds,
                          held_induced_seeds, n_knots, baseline):
    """生成完整 Markdown 报告，包含两 q 结果和 McNemar 表。"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def fmt_eval(r, label):
        lines = []
        lines.append(f"### run_q={r['run_q']:.2f} ({label})\n")
        lines.append("| 方法 | 误报数 | Wilson 95% CI | 检出数 |")
        lines.append("|------|--------|---------------|--------|")
        lines.append(f"| A (static) | {r['fa_A']}/{r['n_normal']} | ({r['ci_A'][0]:.4f}, {r['ci_A'][1]:.4f}) | {r['det_A']}/{r['n_induced']} |")
        lines.append(f"| B (phase) | {r['fa_B']}/{r['n_normal']} | ({r['ci_B'][0]:.4f}, {r['ci_B'][1]:.4f}) | {r['det_B']}/{r['n_induced']} |")
        lines.append(f"| C (persistent) | {r['fa_C']}/{r['n_normal']} | ({r['ci_C'][0]:.4f}, {r['ci_C'][1]:.4f}) | {r['det_C']}/{r['n_induced']} |")
        lines.append("")

        lines.append("#### Exact McNemar paired comparison\n")
        lines.append("| 比较 | n01 | n10 | n_discordant | p_value |")
        lines.append("|------|-----|-----|-------------|---------|")
        for key, val in r["mcnemar"].items():
            a, b = key.split("_vs_")
            lines.append(f"| {a} vs {b} | {val['n01']} | {val['n10']} | {val['n_discordant']} | {val['p_value']:.6f} |")
        lines.append("")
        return lines

    lines = [
        "# Phase Baseline Evaluation Report (v0.7.5)\n",
        f"dev_normal: {format_seed_list(dev_normal_seeds)}",
        f"held_normal: {format_seed_list(held_normal_seeds)}",
        f"held_induced: {format_seed_list(held_induced_seeds)}",
        f"n_knots: {n_knots}",
        "",
        "---\n",
    ]
    lines.extend(fmt_eval(primary, "主结果"))
    lines.extend(fmt_eval(sensitivity, "sensitivity"))

    # 基线摘要
    lines.append("---\n")
    lines.append("## Phase Baseline 摘要\n")
    all_layers = layer_names_from_baseline(baseline)
    lines.append(f"Layers: {len(all_layers)}, Knots: {baseline['n_knots']}\n")
    lines.append("| Layer | Floor (raw MAD_scaled) |")
    lines.append("|-------|------------------------|")
    for ln in all_layers:
        lines.append(f"| {ln} | {baseline['layer_floor'].get(ln, 0.0):.4f} |")
    lines.append("")

    # 诊断
    lines.append("---\n")
    lines.append("## 诊断信息\n")
    lines.append(f"Primary q={primary['run_q']:.2f}: FA_A={primary['fa_A']}, FA_B={primary['fa_B']}, FA_C={primary['fa_C']}")
    lines.append(f"Sensitivity q={sensitivity['run_q']:.2f}: FA_A={sensitivity['fa_A']}, FA_B={sensitivity['fa_B']}, FA_C={sensitivity['fa_C']}")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ==============================================================================
# CLI
# ==============================================================================

def main():
    """v0.7.5 CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="Phase-aware baseline deviation detection (v0.7.5 frozen)")
    parser.add_argument("data_dir", help="run CSV 根目录")
    parser.add_argument("--dev-normal", default="42-53",
                        help="dev normal seed 范围 (default: 42-53)")
    parser.add_argument("--held-normal", default="54-73",
                        help="held-out normal seed 范围 (default: 54-73)")
    parser.add_argument("--dev-induced", default="42-44",
                        help="dev induced seed 范围 (default: 42-44)")
    parser.add_argument("--held-induced", default="45-47",
                        help="held-out induced seed 范围 (default: 45-47)")
    parser.add_argument("--n-knots", type=int, default=8,
                        help="phase knots 数量 (default: 8)")
    parser.add_argument("--run-q", type=float, default=0.95,
                        help="主结果 run-level 分位 (default: 0.95)")
    parser.add_argument("--sensitivity-q", type=float, default=0.90,
                        help="sensitivity 分位 (default: 0.90)")
    parser.add_argument("--percentile", type=float, default=99,
                        help="静态方法 collapse_score 分位 (default: 99)")
    parser.add_argument("--output", default="report_phase.md",
                        help="输出 Markdown 报告路径 (default: report_phase.md)")
    args = parser.parse_args()

    dev_normal_seeds = parse_seed_range(args.dev_normal)
    dev_induced_seeds = parse_seed_range(args.dev_induced)
    held_normal_seeds = parse_seed_range(args.held_normal)
    held_induced_seeds = parse_seed_range(args.held_induced)

    print("加载数据...")
    runs = load_runs(args.data_dir)
    print(f"  总 runs: {len(runs)}")

    print("输入完整性闸门...")
    validate_inputs(runs, dev_normal_seeds, held_normal_seeds,
                    dev_induced_seeds, held_induced_seeds)
    print("  通过")

    all_seeds = dev_normal_seeds + held_normal_seeds + held_induced_seeds
    if all_seeds:
        target_count = len(set(all_seeds))
        matched = sum(1 for s in set(r["seed"] for r in runs) if s in set(all_seeds))
        print(f"  seed 覆盖: {matched}/{target_count}")

    print("构建基线...")
    thresholds_A = compute_static_scores_per_layer(runs, dev_normal_seeds)
    baseline = phase_baseline(runs, dev_normal_seeds, args.n_knots)
    scores_dev = compute_run_scores_phase(runs, dev_normal_seeds, baseline, "normal")

    # 方法 B/C 阈值
    dev_run_scores_B = sorted(scores_dev[s][0] for s in dev_normal_seeds if s in scores_dev)
    run_threshold_B = np_percentile(dev_run_scores_B, args.run_q * 100)

    # 方法 C 与 B 共用 run_threshold
    run_threshold_C = run_threshold_B

    print(f"  Knots: {args.n_knots}, q={args.run_q:.2f}, "
          f"threshold_B={run_threshold_B:.2f}")

    print("评估...")
    primary = evaluate(runs, dev_normal_seeds, held_normal_seeds, held_induced_seeds,
                       baseline, thresholds_A, run_threshold_B, run_threshold_C, args.run_q)
    sensitivity = evaluate(runs, dev_normal_seeds, held_normal_seeds, held_induced_seeds,
                           baseline, thresholds_A, run_threshold_B, run_threshold_C, args.sensitivity_q)

    # 输出摘要
    ci_A = primary["ci_A"]
    ci_B = primary["ci_B"]
    ci_C = primary["ci_C"]
    print(f"  [Primary q={args.run_q:.2f}] "
          f"FA: A={primary['fa_A']}, B={primary['fa_B']}, C={primary['fa_C']}; "
          f"Det: A={primary['det_A']}, B={primary['det_B']}, C={primary['det_C']}")
    print(f"    Wilson CI: A=({ci_A[0]:.4f},{ci_A[1]:.4f}), "
          f"B=({ci_B[0]:.4f},{ci_B[1]:.4f}), C=({ci_C[0]:.4f},{ci_C[1]:.4f})")
    for key, val in primary["mcnemar"].items():
        a, b = key.split("_vs_")
        print(f"    McNemar {a} vs {b}: n01={val['n01']}, n10={val['n10']}, "
              f"n_discordant={val['n_discordant']}, p={val['p_value']:.6f}")

    ci_A_s = sensitivity["ci_A"]
    ci_B_s = sensitivity["ci_B"]
    ci_C_s = sensitivity["ci_C"]
    print(f"  [Sensitivity q={args.sensitivity_q:.2f}] "
          f"FA: A={sensitivity['fa_A']}, B={sensitivity['fa_B']}, C={sensitivity['fa_C']}; "
          f"Det: A={sensitivity['det_A']}, B={sensitivity['det_B']}, C={sensitivity['det_C']}")
    print(f"    Wilson CI: A=({ci_A_s[0]:.4f},{ci_A_s[1]:.4f}), "
          f"B=({ci_B_s[0]:.4f},{ci_B_s[1]:.4f}), C=({ci_C_s[0]:.4f},{ci_C_s[1]:.4f})")
    for key, val in sensitivity["mcnemar"].items():
        a, b = key.split("_vs_")
        print(f"    McNemar {a} vs {b}: n01={val['n01']}, n10={val['n10']}, "
              f"n_discordant={val['n_discordant']}, p={val['p_value']:.6f}")

    # 诊断
    print("\n诊断:")
    all_layers = layer_names_from_baseline(baseline)
    for ln in all_layers:
        floor_val = baseline["layer_floor"].get(ln, 0.0)
        med_vals = [k["median"].get(ln, 0.0) for k in baseline["knots"]]
        mad_vals = [k["mad"].get(ln, 0.0) for k in baseline["knots"]]
        print(f"  {ln}: floor={floor_val:.4f}, "
              f"median_range=[{min(med_vals):.4f}, {max(med_vals):.4f}], "
              f"mad_range=[{min(mad_vals):.4f}, {max(mad_vals):.4f}]")

    write_markdown_report(args.output, primary, sensitivity, dev_normal_seeds,
                          held_normal_seeds, held_induced_seeds, args.n_knots, baseline)
    print(f"\n报告已写入 {args.output}")


if __name__ == "__main__":
    main()

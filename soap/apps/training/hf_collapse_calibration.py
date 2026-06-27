"""
SOAP-Core: HF Collapse Calibration
读取 hf_repr_experiment.py 产出的逐层几何特征 CSV，进行 leave-one-seed-out 校准 + 双规则检测 + 误报率分析 + 传播路径分析。
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_runs(input_dir: str) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    """
    读取目录下所有匹配 hf_repr_seed*_*.csv 的文件。

    Returns:
        dict: {(seed:int, condition:str): [row_dict, ...]}
              row_dict 包含各列，数值列已转为 float/int
    """
    runs: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    pattern = os.path.join(input_dir, "hf_repr_seed*_*.csv")

    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        # 从文件名提取 seed 和 condition（文件名格式：hf_repr_seed42_normal.csv）
        match = re.match(r"hf_repr_seed(\d+)_(.+)\.csv", filename)
        if not match:
            continue

        seed = int(match.group(1))
        condition = match.group(2)

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, Any]] = []
            for row in reader:
                # 数值列转换
                row["seed"] = int(row["seed"])
                row["timestamp"] = row["timestamp"].strip()
                row["layer_name"] = row["layer_name"].strip()
                row["loss"] = float(row["loss"])
                row["val_loss"] = float(row["val_loss"])
                row["grad_norm"] = float(row["grad_norm"])
                row["learning_rate"] = float(row["learning_rate"])
                row["effective_rank"] = float(row["effective_rank"])
                row["representation_variance"] = float(row["representation_variance"])
                row["collapse_score"] = float(row["collapse_score"])
                rows.append(row)
            runs[(seed, condition)] = rows

    return runs


def leave_one_seed_out_fpr(
    runs: Dict[Tuple[int, str], List[Dict[str, Any]]],
    percentile: float = 99,
) -> Dict[str, Any]:
    """
    对每个 normal seed 做 leave-one-seed-out 校准，计算 any_layer 和 persistent 误报率。

    Returns:
        dict with keys: thresholds, any_layer_fpr, persistent_fpr, per_layer_fpr
    """
    # 获取所有 normal seed
    normal_seeds = sorted([seed for (seed, cond) in runs.keys() if cond == "normal"])
    assert len(normal_seeds) >= 3, f"需要至少 3 个 normal seed，当前只有 {normal_seeds}"

    # 提取所有层名
    all_layers = sorted({r["layer_name"] for rows in runs.values() for r in rows})

    # 收集每层的 all_scores（用于最终阈值计算）
    all_normal_scores: Dict[str, List[float]] = defaultdict(list)
    for (seed, cond), rows in runs.items():
        if cond == "normal":
            for r in rows:
                all_normal_scores[r["layer_name"]].append(r["collapse_score"])

    # 每折的检测结果
    any_layer_folds: List[float] = []
    persistent_folds: List[float] = []
    per_layer_folds: Dict[str, List[float]] = defaultdict(list)

    for held_out_seed in normal_seeds:
        # 校准种子：除 held_out_seed 外的 normal seeds
        calibration_seeds = [s for s in normal_seeds if s != held_out_seed]

        # 从校准种子逐层收集 collapse_score，计算阈值
        cal_scores: Dict[str, List[float]] = defaultdict(list)
        for s in calibration_seeds:
            key = (s, "normal")
            if key in runs:
                for r in runs[key]:
                    cal_scores[r["layer_name"]].append(r["collapse_score"])

        # 逐层计算阈值
        thresholds: Dict[str, float] = {}
        for layer_name in all_layers:
            if layer_name in cal_scores and len(cal_scores[layer_name]) > 0:
                thresholds[layer_name] = float(
                    np.percentile(cal_scores[layer_name], percentile)
                )
            else:
                thresholds[layer_name] = float("inf")

        # 获取 held-out seed 的 normal 数据，按层和 checkpoint 时间排序
        held_out_key = (held_out_seed, "normal")
        if held_out_key not in runs:
            continue

        held_out_rows = runs[held_out_key]
        # 按层分组
        layer_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in held_out_rows:
            layer_data[r["layer_name"]].append(r)
        # 按 timestamp 排序
        for layer_name in layer_data:
            layer_data[layer_name].sort(key=lambda x: x["timestamp"])

        # ---- any_layer 规则检测 ----
        any_layer_detected = False
        per_layer_detected: Dict[str, bool] = {ln: False for ln in all_layers}
        for layer_name in all_layers:
            if layer_name in layer_data:
                for r in layer_data[layer_name]:
                    if r["collapse_score"] > thresholds.get(layer_name, float("inf")):
                        any_layer_detected = True
                        per_layer_detected[layer_name] = True

        # ---- persistent 规则检测（连续 >=3 个 checkpoint 超阈值）----
        persistent_detected = False
        for layer_name in all_layers:
            if layer_name in layer_data:
                consecutive_count = 0
                for r in layer_data[layer_name]:
                    if r["collapse_score"] > thresholds.get(layer_name, float("inf")):
                        consecutive_count += 1
                        if consecutive_count >= 3:
                            persistent_detected = True
                            break
                    else:
                        consecutive_count = 0
            if persistent_detected:
                break

        # 记录误报（normal seed 上检出即为误报）
        any_layer_folds.append(1.0 if any_layer_detected else 0.0)
        persistent_folds.append(1.0 if persistent_detected else 0.0)
        for layer_name in all_layers:
            per_layer_folds[layer_name].append(
                1.0 if per_layer_detected[layer_name] else 0.0
            )

    # 计算最终阈值（用全部 3 个 normal seed 的数据）
    final_thresholds: Dict[str, float] = {}
    for layer_name in all_layers:
        if layer_name in all_normal_scores and len(all_normal_scores[layer_name]) > 0:
            final_thresholds[layer_name] = float(
                np.percentile(all_normal_scores[layer_name], percentile)
            )
        else:
            final_thresholds[layer_name] = float("inf")

    # 计算平均误报率
    any_layer_fpr = float(np.mean(any_layer_folds)) if any_layer_folds else 0.0
    persistent_fpr = float(np.mean(persistent_folds)) if persistent_folds else 0.0
    per_layer_fpr: Dict[str, float] = {}
    for layer_name in all_layers:
        if layer_name in per_layer_folds and per_layer_folds[layer_name]:
            per_layer_fpr[layer_name] = float(np.mean(per_layer_folds[layer_name]))
        else:
            per_layer_fpr[layer_name] = 0.0

    return {
        "thresholds": final_thresholds,
        "any_layer_fpr": any_layer_fpr,
        "persistent_fpr": persistent_fpr,
        "per_layer_fpr": per_layer_fpr,
    }


def detect_condition_b(
    runs: Dict[Tuple[int, str], List[Dict[str, Any]]],
    thresholds: Dict[str, float],
    persistent_k: int = 3,
) -> Dict[int, Dict[str, Any]]:
    """
    用 normal 校准的 thresholds 对每个 condition_b seed 进行检测。

    Returns:
        dict: {seed: detection_result}
    """
    results: Dict[int, Dict[str, Any]] = {}
    condition_b_seeds = sorted(
        [seed for (seed, cond) in runs.keys() if cond == "condition_b"]
    )

    for seed in condition_b_seeds:
        key = (seed, "condition_b")
        if key not in runs:
            continue

        rows = runs[key]

        # 按层分组，按时间排序
        layer_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            layer_data[r["layer_name"]].append(r)
        for layer_name in layer_data:
            layer_data[layer_name].sort(key=lambda x: x["timestamp"])

        # ---- any_layer 规则 ----
        any_layer_detected = False
        for layer_name, layer_rows in layer_data.items():
            threshold = thresholds.get(layer_name, float("inf"))
            for r in layer_rows:
                if r["collapse_score"] > threshold:
                    any_layer_detected = True
                    break
            if any_layer_detected:
                break

        # ---- persistent 规则 ----
        persistent_detected = False
        for layer_name, layer_rows in layer_data.items():
            threshold = thresholds.get(layer_name, float("inf"))
            consecutive_count = 0
            for r in layer_rows:
                if r["collapse_score"] > threshold:
                    consecutive_count += 1
                    if consecutive_count >= persistent_k:
                        persistent_detected = True
                        break
                else:
                    consecutive_count = 0
            if persistent_detected:
                break

        # ---- collapse 首现层 ----
        # 收集所有超阈值事件，按 timestamp 排序
        collapse_events: List[Tuple[str, str, float]] = []
        for layer_name, layer_rows in layer_data.items():
            threshold = thresholds.get(layer_name, float("inf"))
            for r in layer_rows:
                if r["collapse_score"] > threshold:
                    collapse_events.append(
                        (layer_name, r["timestamp"], r["collapse_score"])
                    )
        collapse_events.sort(key=lambda x: x[1])  # 按 timestamp 排序

        first_layer: Optional[str] = None
        first_timestamp: Optional[str] = None
        propagation_order: Optional[List[str]] = None

        if collapse_events:
            first_layer = collapse_events[0][0]
            first_timestamp = collapse_events[0][1]

            # 传播顺序：按首现时间列出后续也出现超阈值的层
            # 用时间戳分组：同一时间戳的层视为同时触发
            timestamp_layers: Dict[str, List[str]] = defaultdict(list)
            for layer_name, ts, _ in collapse_events:
                if layer_name not in timestamp_layers[ts]:
                    timestamp_layers[ts].append(layer_name)

            # 按时间戳排序
            sorted_timestamps = sorted(timestamp_layers.keys())
            propagation: List[str] = []
            layer_order = [f"layer_{i}" for i in range(7)]
            for ts in sorted_timestamps:
                # 按层顺序排序同一时间戳的层
                ts_layers = sorted(timestamp_layers[ts], key=lambda x: layer_order.index(x) if x in layer_order else 999)
                for ln in ts_layers:
                    if ln not in propagation:
                        propagation.append(ln)

            # 移除首现层
            if first_layer in propagation:
                propagation.remove(first_layer)
            propagation_order = propagation if propagation else None

        # 各层 collapse_score 汇总
        layer_scores: Dict[str, float] = {}
        for layer_name, layer_rows in layer_data.items():
            # 取最大 collapse_score
            layer_scores[layer_name] = max(
                (r["collapse_score"] for r in layer_rows), default=0.0
            )

        results[seed] = {
            "any_layer_detected": any_layer_detected,
            "persistent_detected": persistent_detected,
            "first_layer": first_layer,
            "first_timestamp": first_timestamp,
            "propagation_order": propagation_order,
            "layer_scores": layer_scores,
            "thresholds": thresholds,
        }

    return results


def generate_markdown_report(
    percentile: float,
    persistent_k: int,
    calib_result: Dict[str, Any],
    detection_results: Dict[int, Dict[str, Any]],
) -> str:
    """生成 Markdown 格式的报告。"""
    lines: List[str] = []
    lines.append("# HF Collapse Calibration Report")
    lines.append("")
    lines.append("> 证据边界：3 seed leave-one-seed-out 小样本，FPR 0.333 = 1/3 held-out normal seed 误报，**不外推**为总体百分比。检测对象为预训练 DistilBERT 合成任务 run；induced 为 controlled rank-1 intervention，**不代表自然 mode collapse 已验证**。三种 FPR（per-layer / run-level any_layer / persistent）含义不同，不可混用。")
    lines.append("")
    lines.append(f"- Percentile: {percentile}")
    lines.append(f"- Persistent k: {persistent_k}")
    lines.append("")

    # 阈值表
    lines.append("## Layer Thresholds")
    lines.append("")
    lines.append("| Layer | Threshold |")
    lines.append("|-------|-----------|")
    for layer_name, threshold in sorted(calib_result["thresholds"].items()):
        lines.append(f"| {layer_name} | {threshold:.6f} |")
    lines.append("")

    # FPR 结果
    lines.append("## False Positive Rate (Normal Data)")
    lines.append("")
    lines.append(f"- **run-level any_layer FPR**: {calib_result['any_layer_fpr']:.3f}  (3 seed 小样本，0.333 = 1/3 held-out seed 误报，不外推)")
    lines.append(f"- **persistent FPR**: {calib_result['persistent_fpr']:.3f}  (同左小样本说明)")
    lines.append("")
    lines.append("| Layer | Per-layer FPR |")
    lines.append("|-------|---------------|")
    for layer_name, fpr in sorted(calib_result["per_layer_fpr"].items()):
        lines.append(f"| {layer_name} | {fpr:.3f} |")
    lines.append("")

    # Condition B 检测
    lines.append("## Condition B Detection")
    lines.append("")
    lines.append("| Seed | any_layer | persistent | First Collapse Layer | First Timestamp | Propagation Order |")
    lines.append("|------|-----------|------------|---------------------|-----------------|-------------------|")

    for seed in sorted(detection_results.keys()):
        dr = detection_results[seed]
        any_str = "✓" if dr["any_layer_detected"] else "✗"
        persist_str = "✓" if dr["persistent_detected"] else "✗"
        first_layer = dr["first_layer"] or "-"
        first_ts = dr["first_timestamp"] or "-"
        prop_str = " → ".join(dr["propagation_order"]) if dr["propagation_order"] else "-"
        lines.append(
            f"| {seed} | {any_str} | {persist_str} | {first_layer} | {first_ts} | {prop_str} |"
        )
    lines.append("")

    # 每个 condition_b seed 的详细信息
    lines.append("## Detailed Results per Seed")
    for seed in sorted(detection_results.keys()):
        dr = detection_results[seed]
        lines.append("")
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append("| Layer | Collapse Score | Threshold | Detected |")
        lines.append("|-------|---------------|-----------|----------|")
        for layer_name in sorted(dr["layer_scores"].keys()):
            score = dr["layer_scores"][layer_name]
            threshold = dr["thresholds"].get(layer_name, float("inf"))
            detected = "✓" if score > threshold else "✗"
            lines.append(f"| {layer_name} | {score:.6f} | {threshold:.6f} | {detected} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """主函数，解析命令行参数并执行校准流程。"""
    parser = argparse.ArgumentParser(
        description="HF Collapse Calibration: Leave-one-seed-out 校准 + 双规则检测 + 误报率分析 + 传播路径"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="包含 hf_repr_seed*.csv 文件的目录"
    )
    parser.add_argument(
        "--percentile", type=int, default=99, help="用于计算阈值的百分位数（默认 99）"
    )
    parser.add_argument(
        "--persistent-k", type=int, default=3, help="persistent 规则的连续 checkpoint 数（默认 3）"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="输出 Markdown 报告的文件路径（可选）"
    )
    args = parser.parse_args()

    # ---- 加载数据 ----
    runs = load_runs(args.input_dir)
    print(f"已加载 {len(runs)} 个 run: {sorted(runs.keys())}")

    # ---- Leave-one-seed-out 校准 ----
    calib_result = leave_one_seed_out_fpr(runs, percentile=args.percentile)

    # ---- Condition B 检测 ----
    detection_results = detect_condition_b(
        runs, calib_result["thresholds"], persistent_k=args.persistent_k
    )

    # ---- 控制台输出 ----
    print()
    print("=" * 60)
    print("校准阈值 (基于全部 normal 数据)")
    print("=" * 60)
    for layer_name, threshold in sorted(calib_result["thresholds"].items()):
        print(f"  {layer_name}: {threshold:.6f}")

    print()
    print("=" * 60)
    print("误报率分析 (Leave-One-Seed-Out)")
    print("=" * 60)
    print(f"  Per-layer FPR:")
    for layer_name, fpr in sorted(calib_result["per_layer_fpr"].items()):
        print(f"    {layer_name}: {fpr:.3f}")
    print(f"  any_layer 平均 FPR:  {calib_result['any_layer_fpr']:.3f}")
    print(f"  persistent 平均 FPR: {calib_result['persistent_fpr']:.3f}")

    print()
    print("=" * 60)
    print("Condition B 检测结果")
    print("=" * 60)
    for seed in sorted(detection_results.keys()):
        dr = detection_results[seed]
        any_str = "检出" if dr["any_layer_detected"] else "未检出"
        persist_str = "检出" if dr["persistent_detected"] else "未检出"
        print(f"  Seed {seed}: any_layer={any_str}, persistent={persist_str}")
        if dr["first_layer"] is not None:
            print(f"    首现层: {dr['first_layer']} @ {dr['first_timestamp']}")
        if dr["propagation_order"] is not None:
            print(f"    传播顺序: {' -> '.join(dr['propagation_order'])}")

        # 各层详情
        print(f"    各层 collapse_score vs 阈值:")
        for layer_name in sorted(dr["layer_scores"].keys()):
            score = dr["layer_scores"][layer_name]
            threshold = dr["thresholds"].get(layer_name, float("inf"))
            detected = ">>> DETECTED" if score > threshold else ""
            print(f"      {layer_name}: score={score:.6f}, threshold={threshold:.6f} {detected}")
        print()

    # ---- 输出 MD 报告 ----
    if args.output:
        md_content = generate_markdown_report(
            args.percentile, args.persistent_k, calib_result, detection_results
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"报告已写入: {args.output}")


if __name__ == "__main__":
    main()

"""
Class-conditional geometry 实验结果分析脚本

读取 examples/cc_seed*_*.csv 文件，按 seed 和 condition 分组为
dev_normal / held_normal / dev_induced / held_induced，
对末段 5 个 checkpoint 聚合各层几何指标，输出对比表和方向判定。

入口：python -m soap.apps.training.classcond_analysis [--input-dir DIR]

注意：阈值/参考只用 dev_normal(42-53)；held_normal/held_induced 仅评估。
不要求成功区分良性/病态；方向判定只标箭头，不自动下结论。
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
METRICS = [
    "nc1_ratio",
    "centroid_effective_rank",
    "fisher_ratio",
    "etf_deviation",
    "validation_accuracy",
]

METRIC_CN = {
    "nc1_ratio":            "NC1 Ratio",
    "centroid_effective_rank": "Centroid Eff Rank",
    "fisher_ratio":         "Fisher Ratio",
    "etf_deviation":        "ETF Deviation",
    "validation_accuracy":  "Val Accuracy",
}

LAYERS = [f"layer_{i}" for i in range(7)]

TAIL_SIZE = 5  # 末段 checkpoint 数量
N_LAYERS = 7   # 每个 checkpoint 的层数（layer_0..6）


# ---------------------------------------------------------------------------
# 分组
# ---------------------------------------------------------------------------
def assign_run_group(row):
    """根据 seed 和 condition 列标记 run_group"""
    seed = int(row["seed"])
    cond = str(row["condition"]).strip()
    if 42 <= seed <= 53 and cond == "normal":
        return "dev_normal"
    if 54 <= seed <= 73 and cond == "normal":
        return "held_normal"
    if 42 <= seed <= 44 and cond == "condition_b":
        return "dev_induced"
    if 45 <= seed <= 47 and cond == "condition_b":
        return "held_induced"
    return None


# ---------------------------------------------------------------------------
# 数据加载与预处理
# ---------------------------------------------------------------------------
def load_data(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "cc_seed*_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"错误：在 {input_dir} 下未找到匹配 cc_seed*_*.csv 的文件", file=sys.stderr)
        sys.exit(1)
    print(f"找到 {len(files)} 个 CSV 文件")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """标记 run_group、按 timestamp 排序、分配 ckpt_idx（checkpoint 序号，非行号）"""
    df["run_group"] = df.apply(assign_run_group, axis=1)
    df = df.dropna(subset=["run_group"]).copy()

    df = df.sort_values(["seed", "condition", "timestamp"]).reset_index(drop=True)
    # 每个 checkpoint 占 7 行（layer_0..6），cumcount // 7 得 checkpoint 序号 0..39
    df["ckpt_idx"] = df.groupby(["seed", "condition"]).cumcount() // N_LAYERS

    # 每个 run 应有 7 层 × 40 ckpt = 280 行
    sizes = df.groupby(["seed", "condition"]).size()
    print(f"每个 run 行数: min={sizes.min()}, max={sizes.max()}, n_runs={len(sizes)}")

    max_ckpt = int(df["ckpt_idx"].max())
    print(f"checkpoint 编号范围: 0 … {max_ckpt}（共 {max_ckpt + 1} 个，预期 40）")
    return df


# ---------------------------------------------------------------------------
# 末段聚合
# ---------------------------------------------------------------------------
def compute_tail_per_run(df: pd.DataFrame, tail_size: int = TAIL_SIZE) -> pd.DataFrame:
    """
    对每个 (seed, condition, layer_name) 取末段 tail_size 个 checkpoint，
    计算各指标的均值 → 得到每个 run 的末段代表值。
    """
    max_ckpt = int(df["ckpt_idx"].max())
    tail_start = max_ckpt - tail_size + 1
    df_tail = df[df["ckpt_idx"] >= tail_start]

    tail_per_run = (
        df_tail
        .groupby(["run_group", "seed", "condition", "layer_name"])[METRICS]
        .mean()
        .reset_index()
    )
    return tail_per_run


# ---------------------------------------------------------------------------
# 格式化工具
# ---------------------------------------------------------------------------
def fmt_ms(mean_val: float, std_val: float, width: int = 22) -> str:
    """格式化 mean±std，右对齐到指定宽度"""
    if pd.isna(std_val) or std_val == 0.0:
        s = f"{mean_val:.4f}"
    else:
        s = f"{mean_val:.4f}±{std_val:.4f}"
    return f"{s:>{width}s}"


def fmt_val(val: float, width: int = 18) -> str:
    if pd.isna(val):
        return f"{'N/A':>{width}s}"
    return f"{val:>{width}.4f}"


def direction_arrow(metric: str, dn_mean: float, ind_mean: float, dn_std: float):
    """返回方向标注字符串"""
    if dn_mean == 0:
        pct = 0.0
    else:
        pct = (ind_mean - dn_mean) / abs(dn_mean) * 100.0

    if metric == "nc1_ratio":
        if pct < -3:
            return f"↓ {abs(pct):.1f}%（类内收缩）"
        elif pct > 3:
            return f"↑ {pct:.1f}%（类内松弛）"
        else:
            return f"持平 ({pct:+.1f}%)"
    else:
        if pct < -3:
            return f"↓ {abs(pct):.1f}%"
        elif pct > 3:
            return f"↑ {pct:.1f}%"
        else:
            return f"持平 ({pct:+.1f}%)"


# ---------------------------------------------------------------------------
# 输出：逐层汇总表
# ---------------------------------------------------------------------------
def print_layer_tables(tail_per_run: pd.DataFrame):
    """生成逐层汇总表，返回 (output_lines, dev_mean, dev_std)"""
    lines = []

    dev_ref = tail_per_run[tail_per_run["run_group"] == "dev_normal"]
    hn_ref  = tail_per_run[tail_per_run["run_group"] == "held_normal"]
    di_ref  = tail_per_run[tail_per_run["run_group"] == "dev_induced"]
    hi_ref  = tail_per_run[tail_per_run["run_group"] == "held_induced"]

    dev_mean = dev_ref.groupby("layer_name")[METRICS].mean()
    dev_std  = dev_ref.groupby("layer_name")[METRICS].std()
    hn_mean  = hn_ref.groupby("layer_name")[METRICS].mean()
    di_mean  = di_ref.groupby("layer_name")[METRICS].mean()
    hi_mean  = hi_ref.groupby("layer_name")[METRICS].mean()

    for layer in LAYERS:
        lines.append("")
        lines.append(f"{'=' * 105}")
        lines.append(f"层: {layer}")
        lines.append(f"{'=' * 105}")

        hdr = (
            f"{'指标':<25s} | "
            f"{'dev_normal(mean±std)':>22s} | "
            f"{'held_normal(mean)':>18s} | "
            f"{'dev_induced(mean)':>18s} | "
            f"{'held_induced(mean)':>18s}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for m in METRICS:
            name = METRIC_CN[m]
            dn_m = dev_mean.loc[layer, m] if layer in dev_mean.index else 0.0
            dn_s = dev_std.loc[layer, m]  if layer in dev_std.index  else 0.0
            hn_v = hn_mean.loc[layer, m]  if layer in hn_mean.index  else float("nan")
            di_v = di_mean.loc[layer, m]  if layer in di_mean.index  else float("nan")
            hi_v = hi_mean.loc[layer, m]  if layer in hi_mean.index  else float("nan")

            line = (
                f"{name:<25s} | "
                f"{fmt_ms(dn_m, dn_s)} | "
                f"{fmt_val(hn_v)} | "
                f"{fmt_val(di_v)} | "
                f"{fmt_val(hi_v)}"
            )
            lines.append(line)

    return lines, dev_mean, dev_std


# ---------------------------------------------------------------------------
# 输出：方向判定
# ---------------------------------------------------------------------------
def print_direction_judgment(tail_per_run: pd.DataFrame, dev_mean, dev_std) -> list:
    """induced 合并 vs dev_normal，逐层逐指标标注方向"""
    lines = []
    lines.append("")
    lines.append(f"{'=' * 105}")
    lines.append("方向判定：Induced（dev_induced + held_induced 合并，共 6 seeds）vs dev_normal")
    lines.append(f"{'=' * 105}")

    induced = tail_per_run[tail_per_run["run_group"].isin(["dev_induced", "held_induced"])]
    ind_mean = induced.groupby("layer_name")[METRICS].mean()

    for layer in LAYERS:
        lines.append(f"\n--- {layer} ---")
        for m in METRICS:
            name = METRIC_CN[m]
            dn_m = dev_mean.loc[layer, m]
            ind_m = ind_mean.loc[layer, m] if layer in ind_mean.index else 0.0
            arrow = direction_arrow(m, dn_m, ind_m, dev_std.loc[layer, m])
            lines.append(f"  {name:<25s}: {dn_m:.4f} → {ind_m:.4f}  {arrow}")

    return lines


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Class-conditional geometry 实验结果分析"
    )
    parser.add_argument(
        "--input-dir", type=str, default="examples",
        help="CSV 文件所在目录（默认 examples）",
    )
    args = parser.parse_args()

    # 1. 读取 & 预处理
    print(f"正在从 {args.input_dir} 读取 CSV 文件 ...")
    df = load_data(args.input_dir)
    print(f"总行数: {len(df)}")

    df = preprocess(df)

    # 分组统计
    group_runs = (
        df.groupby("run_group")[["seed", "condition"]]
        .apply(lambda g: g.drop_duplicates().shape[0])
    )
    print("\n各组 run 数量：")
    for g in ["dev_normal", "held_normal", "dev_induced", "held_induced"]:
        print(f"  {g}: {group_runs.get(g, 0)} runs")

    # 2. 末段聚合
    tail_per_run = compute_tail_per_run(df, TAIL_SIZE)

    # 3. 组装输出
    output_lines = []
    output_lines.append("Class-conditional geometry 实验分析结果")
    output_lines.append(f"{'=' * 105}")
    output_lines.append(f"末段 checkpoint 数量: {TAIL_SIZE}")
    output_lines.append("")
    output_lines.append("各组 run 数量：")
    for g in ["dev_normal", "held_normal", "dev_induced", "held_induced"]:
        output_lines.append(f"  {g}: {group_runs.get(g, 0)} runs")

    # 逐层汇总表
    output_lines.append("")
    output_lines.append(f"{'#' * 105}")
    output_lines.append("# 各层指标汇总（末段 checkpoint 的 run 间聚合）")
    output_lines.append(f"{'#' * 105}")

    table_lines, dev_mean, dev_std = print_layer_tables(tail_per_run)
    output_lines.extend(table_lines)

    # 方向判定
    dir_lines = print_direction_judgment(tail_per_run, dev_mean, dev_std)
    output_lines.extend(dir_lines)

    # 4. 输出
    full_text = "\n".join(output_lines)
    print("\n" + full_text)

    out_path = os.path.join(args.input_dir, "_classcond_analysis.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\n分析结果已写入: {out_path}")


if __name__ == "__main__":
    main()

"""
SOAP-Core v0.7.1 Representation Geometry Direct Detection Adapter.

This adapter implements direct detection of representation collapse from
repr-enhanced CSV files, bypassing the SOAP analyze pipeline which is
insensitive to mode collapse dynamics. It uses effective_rank, representation_variance,
and collapse_score metrics to determine collapse status.
"""

import csv
import math
from pathlib import Path

METRIC_COLS = ["effective_rank", "representation_variance", "collapse_score"]


def _col_stats(values):
    """Calculate summary statistics for a list of numeric values.

    Args:
        values: Non-empty list of floats.

    Returns:
        Dictionary with keys: start, end, mean, min, max, drop_ratio.
        drop_ratio = (start - end) / start, returns None if start <= 1e-9 (to avoid division by zero).
    """
    n = len(values)
    start, end = values[0], values[-1]
    drop_ratio = (start - end) / start if start > 1e-9 else None
    return {
        "start": start,
        "end": end,
        "mean": sum(values) / n,
        "min": min(values),
        "max": max(values),
        "drop_ratio": drop_ratio,
    }


def summarize_representation_metrics(csv_path):
    """Read a repr-enhanced CSV and return summary statistics for each metric column.

    Args:
        csv_path: Path to the repr-enhanced CSV file.

    Returns:
        Dictionary mapping each metric column name to its summary statistics.

    Raises:
        ValueError: If required metric columns are missing or have no valid numeric values.
    """
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    missing = [c for c in METRIC_COLS if c not in fieldnames]
    if missing:
        raise ValueError(f"repr-enhanced CSV 缺少 representation 指标列: {missing}，可用列: {fieldnames}")

    summary = {}
    for col in METRIC_COLS:
        vals = []
        for row in rows:
            cell = row.get(col, "")
            if cell == "":
                continue
            try:
                v = float(cell)
            except ValueError:
                continue
            if math.isfinite(v):
                vals.append(v)
        if not vals:
            raise ValueError(f"列 '{col}' 无有效数值")
        summary[col] = _col_stats(vals)

    return summary


def detect_representation_collapse(summary, thresholds=None):
    """Detect representation collapse using direct detection rules.

    Args:
        summary: Dictionary from summarize_representation_metrics().
        thresholds: Optional dictionary to override default threshold values.
            Keys: collapse_score_min, effective_rank_max, drop_ratio_min.

    Returns:
        Dictionary with keys: status, is_collapse, collapse_score_final,
        effective_rank_final, effective_rank_drop_ratio, reason.
    """
    th = {
        "collapse_score_min": 0.9,
        "effective_rank_max": 1.5,
        "drop_ratio_min": 0.7,
    }
    if thresholds:
        th.update(thresholds)

    cs_final = summary["collapse_score"]["end"]
    er_final = summary["effective_rank"]["end"]
    er_drop = summary["effective_rank"].get("drop_ratio")

    reasons = []
    if cs_final > th["collapse_score_min"] and er_final < th["effective_rank_max"]:
        reasons.append(
            f"collapse_score_final={cs_final:.3f}>{th['collapse_score_min']}"
            f" 且 effective_rank_final={er_final:.3f}<{th['effective_rank_max']}"
        )
    # 注：曾用 effective_rank drop_ratio>阈值 作辅助规则，实测正常训练（normal/overfit）
    # drop_ratio 也达 0.76~0.79（训练中 eff_rank 自然聚焦下降），误判严重，故移除。
    # collapse 以 final 值（eff_rank 接近 1、collapse_score 高）为准，不以变化比例为准。
    # th 仍保留 drop_ratio_min 键以兼容 thresholds 接口，但不参与判定。

    is_collapse = len(reasons) > 0
    return {
        "status": "collapse" if is_collapse else "ok",
        "is_collapse": is_collapse,
        "collapse_score_final": cs_final,
        "effective_rank_final": er_final,
        "effective_rank_drop_ratio": er_drop,
        "reason": "; ".join(reasons) if reasons else "no collapse signal",
    }


if __name__ == "__main__":
    import sys

    examples = Path(__file__).resolve().parents[3] / "examples"
    targets = ["normal", "divergence", "overfit", "mode_collapse"]
    for name in targets:
        p = examples / f"torch_training_{name}_repr_enhanced.csv"
        if not p.exists():
            print(f"skip (not found): {p}", file=sys.stderr)
            continue
        s = summarize_representation_metrics(p)
        d = detect_representation_collapse(s)
        print(
            f"[{name}] status={d['status']}"
            f"  collapse_score_final={d['collapse_score_final']:.4f}"
            f"  effective_rank_final={d['effective_rank_final']:.4f}"
            f"  drop_ratio={d['effective_rank_drop_ratio']}"
        )

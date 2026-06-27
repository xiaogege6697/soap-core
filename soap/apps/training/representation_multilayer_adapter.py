import csv, math
from pathlib import Path
from soap.apps.training.representation_adapter import _col_stats, detect_representation_collapse, METRIC_COLS

LAYER_COL = "layer_name"

def summarize_multilayer(csv_path):
    """读 multilayer CSV（含 layer_name 列 + METRIC_COLS 三列），按 layer_name 分组，每层每指标算 _col_stats。
    返回 {layer_name: {col: stats}}。缺 layer_name 列或 METRIC_COLS 任一列 raise ValueError。跳过空/非有限值。"""
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if LAYER_COL not in fieldnames:
        raise ValueError(f"多层 CSV 缺少 '{LAYER_COL}' 列，可用列: {fieldnames}")
    missing = [c for c in METRIC_COLS if c not in fieldnames]
    if missing:
        raise ValueError(f"缺少 representation 指标列: {missing}")
    # 按 layer_name 分组
    layers = {}
    for row in rows:
        ln = row.get(LAYER_COL, "")
        layers.setdefault(ln, []).append(row)
    summary = {}
    for ln, ln_rows in layers.items():
        col_stats = {}
        for col in METRIC_COLS:
            vals = []
            for row in ln_rows:
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
                raise ValueError(f"层 '{ln}' 列 '{col}' 无有效数值")
            col_stats[col] = _col_stats(vals)
        summary[ln] = col_stats
    return summary

def detect_multilayer_collapse(csv_path, aggregation="any_layer", thresholds=None, consecutive_min=2):
    """逐层 detect（复用 detect_representation_collapse）+ 三种聚合。
    返回 dict：
      { "per_layer": {layer: <detect_result>},
        "aggregations": {"any_layer": bool, "consecutive": bool, "severity_weighted": bool},
        "severity_mean": float,   # 各层 collapse_score_final 均值
        "layers_in_order": [layer_names 排序后] }
    聚合规则：
      - any_layer: 任一层 is_collapse
      - consecutive: 按 layer_name 排序，存在连续 >= consecutive_min 层都 collapse
      - severity_weighted: 各层 collapse_score_final 均值 > 0.9（保守 proxy）
    逐层诊断在 per_layer 完整输出。"""
    summaries = summarize_multilayer(csv_path)
    per_layer = {}
    for layer, summ in summaries.items():
        per_layer[layer] = detect_representation_collapse(summ, thresholds)
    layers_in_order = sorted(summaries.keys())
    # any_layer
    any_collapse = any(d["is_collapse"] for d in per_layer.values())
    # consecutive：按排序后层顺序，最长连续 collapse 段 >= consecutive_min
    max_run = 0; cur = 0
    for layer in layers_in_order:
        if per_layer[layer]["is_collapse"]:
            cur += 1; max_run = max(max_run, cur)
        else:
            cur = 0
    consecutive = max_run >= consecutive_min
    # severity_weighted
    cs_vals = [per_layer[l]["collapse_score_final"] for l in layers_in_order]
    severity_mean = sum(cs_vals) / len(cs_vals) if cs_vals else 0.0
    severity_weighted = severity_mean > 0.9
    return {
        "per_layer": per_layer,
        "aggregations": {
            "any_layer": any_collapse,
            "consecutive": consecutive,
            "severity_weighted": severity_weighted,
        },
        "severity_mean": severity_mean,
        "layers_in_order": layers_in_order,
    }

if __name__ == "__main__":
    import sys
    examples = Path(__file__).resolve().parents[3] / "examples"
    for name in ["normal", "divergence", "overfit", "mode_collapse"]:
        p = examples / f"torch_training_{name}_multilayer_repr.csv"
        if not p.exists():
            print(f"skip: {p}", file=sys.stderr); continue
        r = detect_multilayer_collapse(p)
        print(f"[{name}] severity_mean={r['severity_mean']:.4f}  aggregations={r['aggregations']}")
        for layer in r["layers_in_order"]:
            d = r["per_layer"][layer]
            print(f"    {layer}: status={d['status']}  collapse_score={d['collapse_score_final']:.4f}  eff_rank={d['effective_rank_final']:.4f}")

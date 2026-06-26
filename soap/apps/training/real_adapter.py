"""真实训练日志 CSV 适配器（v0.6.5）。把外部字段名（W&B/TensorBoard/HuggingFace 风格）映射到 SOAP 标准字段，做最小语义校验，返回与核心 loader 兼容的 TimeSeries。不改核心 loader，复用 load_csv。"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from soap.data.loader import load_csv


STANDARD_FIELDS = ["step", "loss", "val_loss", "grad_norm", "learning_rate"]


REQUIRED_FIELDS = ["loss"]


NON_NEGATIVE_FIELDS = ["loss", "val_loss", "grad_norm", "learning_rate"]


def normalize_training_log(path, field_map=None):
    """读取训练日志 CSV，映射字段，校验语义，返回 TimeSeries（兼容 load_csv）。

    path: CSV 路径（str 或 Path）
    field_map: dict 外部列名->标准字段，如 {"train/loss":"loss","eval/loss":"val_loss","train/grad_norm":"grad_norm","lr":"learning_rate"}。None 时假设已是标准字段名；默认把 timestamp 也当作 step。
    返回：load_csv 兼容的 TimeSeries。
    """
    path = Path(path)

    # 1. 读取 CSV（首行列名 + 数据行）
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not raw_fieldnames:
        raise ValueError(f"CSV 为空或无可读列名: {path}")

    # 2. 列名映射
    effective_map = dict(field_map) if field_map else {}
    # 默认：若无显式映射且存在 timestamp 列但无 step 列，则 timestamp -> step
    if not field_map and "timestamp" in raw_fieldnames and "step" not in raw_fieldnames:
        effective_map["timestamp"] = "step"

    # 构建映射后的列名列表（保持顺序）
    mapped_names = [effective_map.get(col, col) for col in raw_fieldnames]

    # 3. 选出存在的标准字段列（保持 STANDARD_FIELDS 顺序）
    selected_indices = []
    selected_fields = []
    for std in STANDARD_FIELDS:
        if std in mapped_names:
            idx = mapped_names.index(std)
            selected_indices.append(idx)
            selected_fields.append(std)

    # loss 必需
    for req in REQUIRED_FIELDS:
        if req not in selected_fields:
            raise ValueError(f"必需字段 '{req}' 缺失，可用列: {raw_fieldnames}")

    # 4. 数值校验：非负字段 <0 则 raise
    for neg_field in NON_NEGATIVE_FIELDS:
        if neg_field in selected_fields:
            col_idx = selected_fields.index(neg_field)
            raw_col_idx = selected_indices[col_idx]
            raw_col_name = raw_fieldnames[raw_col_idx]
            for row_i, row in enumerate(rows):
                cell = row.get(raw_col_name, "")
                if cell == "":
                    continue
                try:
                    val = float(cell)
                except ValueError:
                    continue
                if val < 0:
                    raise ValueError(
                        f"非负字段 '{neg_field}' 在第 {row_i + 2} 行出现负值: {val}"
                    )

    # 5. step 单调性检查
    if "step" in selected_fields:
        step_col_idx = selected_fields.index("step")
        raw_step_idx = selected_indices[step_col_idx]
        raw_step_name = raw_fieldnames[raw_step_idx]
        numeric_steps = []
        for row in rows:
            cell = row.get(raw_step_name, "")
            try:
                numeric_steps.append(float(cell))
            except (ValueError, TypeError):
                numeric_steps = []
                break

        if len(numeric_steps) >= 2:
            # 严格递减检测
            strictly_decreasing = all(
                numeric_steps[i] > numeric_steps[i + 1]
                for i in range(len(numeric_steps) - 1)
            )
            if strictly_decreasing:
                raise ValueError("step 列严格递减，疑似数据倒序")

            # 非单调递增则 warning
            monotone_inc = all(
                numeric_steps[i] <= numeric_steps[i + 1]
                for i in range(len(numeric_steps) - 1)
            )
            if not monotone_inc:
                print(
                    "Warning: step 列非单调递增，可能存在乱序或重复",
                    file=sys.stderr,
                )

    # 6. 写临时 CSV，调用 load_csv
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", newline="", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
        writer = csv.writer(tmp)
        writer.writerow(selected_fields)
        for row in rows:
            out_row = []
            for raw_idx in selected_indices:
                out_row.append(row.get(raw_fieldnames[raw_idx], ""))
            writer.writerow(out_row)

    return load_csv(tmp_path)


def describe_fields(path, field_map=None) -> dict:
    """返回 {"standard_fields_present": [...], "missing_optional": [...], "rows": int}。"""
    path = Path(path)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not raw_fieldnames:
        return {
            "standard_fields_present": [],
            "missing_optional": [f for f in STANDARD_FIELDS if f not in REQUIRED_FIELDS],
            "rows": 0,
        }

    effective_map = dict(field_map) if field_map else {}
    if not field_map and "timestamp" in raw_fieldnames and "step" not in raw_fieldnames:
        effective_map["timestamp"] = "step"

    mapped_names = {effective_map.get(col, col) for col in raw_fieldnames}

    present = [sf for sf in STANDARD_FIELDS if sf in mapped_names]
    optional = [sf for sf in STANDARD_FIELDS if sf not in REQUIRED_FIELDS]
    missing_optional = [sf for sf in optional if sf not in present]

    return {
        "standard_fields_present": present,
        "missing_optional": missing_optional,
        "rows": len(rows),
    }


if __name__ == "__main__":
    _demo_path = Path("examples/realish_training_wandb.csv")
    _demo_field_map = {
        "train/loss": "loss",
        "eval/loss": "val_loss",
        "train/grad_norm": "grad_norm",
        "lr": "learning_rate",
    }
    if _demo_path.exists():
        _ts = normalize_training_log(_demo_path, _demo_field_map)
        print(_ts)
        print(describe_fields(_demo_path, _demo_field_map))
    else:
        print(f"demo CSV not found: {_demo_path}", file=sys.stderr)

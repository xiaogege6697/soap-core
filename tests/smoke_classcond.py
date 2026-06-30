"""v0.7.6 hf_classcond_experiment 端到端 smoke 断言。

最小真训练 run（CPU，确定性高），验证修复后的脚本：
1. train=1600, val=400, probe=128
2. probe 不进入训练索引（train/val/probe 两两不重叠）
3. CSV 数据行数 = checkpoint 数 × 7
4. 每个 checkpoint 恰好 7 层
5. validation_accuracy 每 checkpoint 唯一且 ∈ [0,1]
6. 无 layer_name="accuracy" 伪层
7. 最小真训练 run 跑通，全指标无 NaN/Inf

运行：.venv/bin/python tests/smoke_classcond.py
文件名无 test_ 前缀，不进默认 pytest 全量（保持单测快速）。

流程教训（2026-06-29）：py_compile + 单元测试 ≠ 主入口真跑过；
以后"审核通过"必须包含一次最小端到端 smoke。
"""
import csv
import os
import subprocess
import sys
from collections import Counter

import numpy as np

from soap.apps.training.hf_repr_experiment import make_structured_dataset


SMOKE_CSV = "examples/_smoke_classcond.csv"
DEVICE = "cpu"  # smoke 用 CPU 保证确定性；可改 auto
STEPS = 6
RECORD_EVERY = 3  # → 2 checkpoint


def check(cond, msg):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def main():
    print("=== 断言1: 数据集三池规模 train=1600, val=400, probe=128 ===")
    texts_train, labels_train, texts_val, labels_val, texts_probe, labels_probe = \
        make_structured_dataset(data_seed=0)
    check(len(texts_train) == 1600, f"train={len(texts_train)}==1600")
    check(len(texts_val) == 400, f"val={len(texts_val)}==400")
    check(len(texts_probe) == 128, f"probe={len(texts_probe)}==128")
    # 类别数从 labels_train 推导，应为 5
    check(max(labels_train) + 1 == 5, f"num_classes={max(labels_train)+1}==5")

    print("=== 断言2: train/val/probe 两两不重叠（probe 不入训练索引）===")
    s_train, s_val, s_probe = set(texts_train), set(texts_val), set(texts_probe)
    check(len(s_train & s_probe) == 0, f"train∩probe={len(s_train & s_probe)}==0")
    check(len(s_train & s_val) == 0, f"train∩val={len(s_train & s_val)}==0")
    check(len(s_val & s_probe) == 0, f"val∩probe={len(s_val & s_probe)}==0")

    print(f"=== 跑最小真训练 run（steps={STEPS}, record_every={RECORD_EVERY}, device={DEVICE}）===")
    if os.path.exists(SMOKE_CSV):
        os.remove(SMOKE_CSV)
    cmd = [
        sys.executable, "-m", "soap.apps.training.hf_classcond_experiment",
        "--seed", "42", "--condition", "normal",
        "--steps", str(STEPS), "--batch-size", "16",
        "--record-every", str(RECORD_EVERY), "--probe-size", "128",
        "--device", DEVICE, "--output", SMOKE_CSV,
    ]
    print(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  STDOUT:", result.stdout[-800:])
        print("  STDERR:", result.stderr[-1500:])
        raise AssertionError(f"实验进程退出码 {result.returncode}（未跑通）")
    check(True, "实验进程正常退出（returncode=0）—— 断言7: 跑通")

    print("=== 断言3: CSV 数据行数 = checkpoint数 × 7 ===")
    with open(SMOKE_CSV) as f:
        rows = list(csv.DictReader(f))
    n_checkpoints = STEPS // RECORD_EVERY
    expected = n_checkpoints * 7
    check(len(rows) == expected, f"CSV 数据行={len(rows)}=={expected}（{n_checkpoints}ckpt×7层）")

    print("=== 断言4: 每个 checkpoint 恰好 7 层 ===")
    ts_counts = Counter(r["timestamp"] for r in rows)
    check(len(ts_counts) == n_checkpoints, f"checkpoint 数={len(ts_counts)}=={n_checkpoints}")
    for ts, cnt in ts_counts.items():
        check(cnt == 7, f"timestamp={ts} 层数={cnt}==7")

    print("=== 断言5: validation_accuracy 每 checkpoint 唯一且 ∈[0,1] ===")
    for ts in ts_counts:
        accs = {round(float(r["validation_accuracy"]), 6)
                for r in rows if r["timestamp"] == ts}
        check(len(accs) == 1, f"timestamp={ts} val_acc 唯一（得 {len(accs)} 个不同值）")
        acc_val = accs.pop()
        check(0.0 <= acc_val <= 1.0, f"timestamp={ts} val_acc={acc_val:.4f}∈[0,1]")

    print("=== 断言6: 无 layer_name='accuracy' 伪层 ===")
    layer_names = {r["layer_name"] for r in rows}
    check("accuracy" not in layer_names, f"无 accuracy 伪层（layer_names={sorted(layer_names)}）")
    check(layer_names == {f"layer_{i}" for i in range(7)},
          f"layer_names={sorted(layer_names)}==layer_0..6")

    print("=== 断言7（续）: 全指标无 NaN/Inf ===")
    float_cols = ["nc1_ratio", "centroid_effective_rank", "fisher_ratio",
                  "etf_deviation", "validation_accuracy"]
    bad = 0
    for r in rows:
        for c in float_cols:
            v = float(r[c])
            if np.isnan(v) or np.isinf(v):
                bad += 1
    check(bad == 0, f"无 NaN/Inf（{bad} 个异常值）")

    print("\n✅ 全部 7 项 smoke 断言通过")
    print(f"smoke CSV: {SMOKE_CSV}（task17 前清理）")


if __name__ == "__main__":
    main()

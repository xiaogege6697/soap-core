"""SOAP-Core v0.6.9 — Enhanced Observation Training Log Generator

基于 v0.6.8 torch_runs 的训练设置（相同 seed 复现动力学），每步额外记录两个观测维度：
- train_val_gap   = val_loss - loss          （攻 overfit：泛化 gap）
- output_variance = 模型对验证集输出方差       （攻 mode_collapse：输出坍缩）
四类统一记录真实 output_variance（非 proxy），可比性一致。
目的：验证扩展观测维度能否解锁 normal/overfit/mode_collapse 区分。

Run from project root:  python -m soap.apps.training.enhanced_runs
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from soap.apps.training.torch_runs import make_dataset, _make_mlp, _safe_float, _grad_norm


HEADER = ["timestamp", "loss", "val_loss", "grad_norm", "learning_rate", "train_val_gap", "output_variance"]


def _run(out_csv, n_train, hidden, optim_fn, steps, seed, batch_size=None):
    """通用 enhanced 训练循环。

    optim_fn(model) -> optimizer；batch_size=None 全批，否则 mini-batch。
    训练设置与 v0.6.8 torch_runs 完全一致，仅多记 train_val_gap / output_variance。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val = make_dataset(n_train, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], hidden)
    optimizer = optim_fn(model)
    criterion = nn.MSELoss()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        last_loss = last_val = last_gn = last_var = 1.0
        for step in range(steps):
            # ---- train ----
            model.train()
            optimizer.zero_grad()
            if batch_size is None:
                pred = model(X_train)
                loss = criterion(pred, y_train)
            else:
                idx = torch.randperm(X_train.shape[0])[:batch_size]
                pred = model(X_train[idx])
                loss = criterion(pred, y_train[idx])
            loss.backward()
            gn = _grad_norm(model)
            optimizer.step()

            # ---- val + output variance ----
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
                out_var = float(torch.var(val_pred).item())

            loss_val = _safe_float(loss.item(), last_loss)
            val_loss = _safe_float(val_loss, last_val)
            gn = _safe_float(gn, last_gn)
            out_var = _safe_float(out_var, last_var)
            lr = optimizer.param_groups[0]["lr"]
            train_val_gap = val_loss - loss_val

            last_loss, last_val, last_gn, last_var = loss_val, val_loss, gn, out_var
            w.writerow([f"step_{step}", loss_val, val_loss, gn, lr, train_val_gap, out_var])
    return out_csv


def run_normal_enhanced(out_csv, steps=300, seed=42):
    return _run(out_csv, 200, [32, 16],
                lambda m: torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4), steps, seed)


def run_divergence_enhanced(out_csv, steps=300, seed=42):
    return _run(out_csv, 200, [32, 16],
                lambda m: torch.optim.SGD(m.parameters(), lr=0.5), steps, seed, batch_size=16)


def run_overfit_enhanced(out_csv, steps=400, seed=42):
    return _run(out_csv, 30, [128, 64],
                lambda m: torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.0), steps, seed)


def run_mode_collapse_enhanced(out_csv, steps=300, seed=42):
    return _run(out_csv, 200, [64, 32],
                lambda m: torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=5.0), steps, seed)


def _summary(csv_path):
    with open(csv_path) as f:
        r = csv.reader(f)
        h = next(r)
        rows = list(r)
    li = h.index("loss")
    vi = h.index("output_variance")
    gi = h.index("train_val_gap")
    print(f"   loss 首/尾={float(rows[0][li]):.4f}/{float(rows[-1][li]):.4f}  "
          f"gap 首/尾={float(rows[0][gi]):.4f}/{float(rows[-1][gi]):.4f}  "
          f"out_var 首/尾={float(rows[0][vi]):.4f}/{float(rows[-1][vi]):.4f}")


def main():
    examples_dir = Path(__file__).resolve().parents[3] / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    for name, fn, steps in [
        ("normal", run_normal_enhanced, 300),
        ("divergence", run_divergence_enhanced, 300),
        ("overfit", run_overfit_enhanced, 400),
        ("mode_collapse", run_mode_collapse_enhanced, 300),
    ]:
        p = examples_dir / f"torch_training_{name}_enhanced.csv"
        fn(str(p), steps=steps)
        print(f"[{name}] → {p}")
        _summary(p)


if __name__ == "__main__":
    main()

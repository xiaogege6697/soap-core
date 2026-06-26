"""
SOAP-Core v0.6.8 — Real PyTorch Training Log Generator

Generates four classes of training logs using actual PyTorch training dynamics
(forward, backward, optimizer step, gradient norms) to validate that the SOAP
instability taxonomy transfers from synthetic curves to real training behaviour.

Run from project root:  python -m soap.apps.training.torch_runs
"""

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared data generation
# ---------------------------------------------------------------------------

def make_dataset(n_train, n_val=200, n_features=6, seed=42):
    """Generate a small non-linear regression dataset entirely in-distribution.

    Returns (X_train, y_train, X_val, y_val) as float32 tensors.
    Target: y = 2*sin(x1) + 0.5*sum(x^2) + small noise
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_val
    X_np = rng.randn(n_total, n_features).astype(np.float32)
    # non-linear target
    y_np = (
        2.0 * np.sin(X_np[:, 0])
        + 0.5 * np.sum(X_np ** 2, axis=1)
        + 0.1 * rng.randn(n_total).astype(np.float32)
    )
    X_train = torch.from_numpy(X_np[:n_train])
    y_train = torch.from_numpy(y_np[:n_train]).unsqueeze(1)
    X_val   = torch.from_numpy(X_np[n_train:])
    y_val   = torch.from_numpy(y_np[n_train:]).unsqueeze(1)
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _make_mlp(n_features, hidden_sizes):
    """Simple MLP with ReLU activations, no activation on output."""
    layers = []
    in_dim = n_features
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value, last_good, cap=1e4):
    """Return a finite Python float, falling back to *last_good* or *cap*."""
    v = float(value)
    if not math.isfinite(v):
        return min(last_good, cap) if last_good is not None else cap
    return min(v, cap)


def _grad_norm(model):
    """Compute L2 norm of all parameter gradients."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data
            total_sq += float(g.norm(2).item()) ** 2
    return math.sqrt(total_sq) if total_sq > 0 else 0.0


def _open_csv(path, header):
    """Open a CSV for writing and write the header row."""
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(header)
    return f, writer


def _close_csv(f):
    f.flush()
    f.close()


# ---------------------------------------------------------------------------
# Run 1: Normal training
# ---------------------------------------------------------------------------

def run_normal(out_csv, steps=300, seed=42):
    """Stable Adam training — smooth loss decline, steady grad norms."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [32, 16])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    header = ["timestamp", "loss", "val_loss", "grad_norm", "learning_rate"]
    f_csv, writer = _open_csv(out_csv, header)

    last_loss = 1.0
    last_val  = 1.0
    last_gn   = 1.0

    for step in range(steps):
        # ---- train ----
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        gn = _grad_norm(model)
        optimizer.step()

        # ---- val ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        loss_val = loss.item()
        lr = optimizer.param_groups[0]["lr"]

        loss_val = _safe_float(loss_val, last_loss)
        val_loss = _safe_float(val_loss, last_val)
        gn       = _safe_float(gn, last_gn)

        last_loss, last_val, last_gn = loss_val, val_loss, gn

        writer.writerow([f"step_{step}", loss_val, val_loss, gn, lr])

    _close_csv(f_csv)


# ---------------------------------------------------------------------------
# Run 2: Divergence
# ---------------------------------------------------------------------------

def run_divergence(out_csv, steps=300, seed=42):
    """Excessively high LR with SGD — loss / grad_norm blow up."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [32, 16])
    batch_size = 16
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.MSELoss()

    header = ["timestamp", "loss", "val_loss", "grad_norm", "learning_rate"]
    f_csv, writer = _open_csv(out_csv, header)

    last_loss = 1.0
    last_val  = 1.0
    last_gn   = 1.0

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        idx = torch.randperm(X_train.shape[0])[:batch_size]
        pred = model(X_train[idx])
        loss = criterion(pred, y_train[idx])
        loss.backward()
        gn = _grad_norm(model)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        loss_val = loss.item()
        lr = optimizer.param_groups[0]["lr"]

        loss_val = _safe_float(loss_val, last_loss)
        val_loss = _safe_float(val_loss, last_val)
        gn       = _safe_float(gn, last_gn)

        last_loss, last_val, last_gn = loss_val, val_loss, gn

        writer.writerow([f"step_{step}", loss_val, val_loss, gn, lr])

    _close_csv(f_csv)


# ---------------------------------------------------------------------------
# Run 3: Overfit
# ---------------------------------------------------------------------------

def run_overfit(out_csv, steps=400, seed=42):
    """Tiny train set + large model + no weight decay → clear generalization gap."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val = make_dataset(30, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [128, 64])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.MSELoss()

    header = ["timestamp", "loss", "val_loss", "grad_norm", "learning_rate"]
    f_csv, writer = _open_csv(out_csv, header)

    last_loss = 1.0
    last_val  = 1.0
    last_gn   = 1.0

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        gn = _grad_norm(model)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        loss_val = loss.item()
        lr = optimizer.param_groups[0]["lr"]

        loss_val = _safe_float(loss_val, last_loss)
        val_loss = _safe_float(val_loss, last_val)
        gn       = _safe_float(gn, last_gn)

        last_loss, last_val, last_gn = loss_val, val_loss, gn

        writer.writerow([f"step_{step}", loss_val, val_loss, gn, lr])

    _close_csv(f_csv)


# ---------------------------------------------------------------------------
# Run 4: Mode collapse proxy (heavy L2 regularisation)
# ---------------------------------------------------------------------------

def run_mode_collapse(out_csv, variance_csv=None, steps=300, seed=42):
    """Extreme weight_decay collapses output variance — proxy for mode collapse."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5.0)
    criterion = nn.MSELoss()

    header = ["timestamp", "loss", "val_loss", "grad_norm", "learning_rate"]
    f_csv, writer = _open_csv(out_csv, header)

    var_file = None
    var_writer = None
    if variance_csv is not None:
        var_file, var_writer = _open_csv(variance_csv, ["timestamp", "output_variance"])

    last_loss = 1.0
    last_val  = 1.0
    last_gn   = 1.0

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        gn = _grad_norm(model)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            out_var  = float(torch.var(val_pred).item())

        loss_val = loss.item()
        lr = optimizer.param_groups[0]["lr"]

        loss_val = _safe_float(loss_val, last_loss)
        val_loss = _safe_float(val_loss, last_val)
        gn       = _safe_float(gn, last_gn)
        out_var  = _safe_float(out_var, 1.0)

        last_loss, last_val, last_gn = loss_val, val_loss, gn

        writer.writerow([f"step_{step}", loss_val, val_loss, gn, lr])
        if var_writer is not None:
            var_writer.writerow([f"step_{step}", out_var])

    _close_csv(f_csv)
    if var_file is not None:
        _close_csv(var_file)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    examples_dir = Path(__file__).resolve().parents[3] / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # --- Normal ---
    p = examples_dir / "torch_training_normal.csv"
    run_normal(str(p))
    print(f"[normal]       → {p}")
    _print_first_last_loss(p)

    # --- Divergence ---
    p = examples_dir / "torch_training_divergence.csv"
    run_divergence(str(p))
    print(f"[divergence]   → {p}")
    _print_first_last_loss(p)

    # --- Overfit ---
    p = examples_dir / "torch_training_overfit.csv"
    run_overfit(str(p))
    print(f"[overfit]      → {p}")
    _print_first_last_loss(p)

    # --- Mode collapse ---
    p = examples_dir / "torch_training_mode_collapse.csv"
    pv = examples_dir / "torch_training_mode_collapse_variance.csv"
    run_mode_collapse(str(p), variance_csv=str(pv))
    print(f"[mode_collapse]→ {p}")
    print(f"[variance]     → {pv}")
    _print_first_last_loss(p)


def _print_first_last_loss(csv_path):
    """Quick summary: first & last loss values from a standard CSV."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if not rows:
        print("   (empty)")
        return
    first = rows[0]
    last  = rows[-1]
    loss_idx = header.index("loss")
    print(f"   loss  step 0: {float(first[loss_idx]):.6f}  |  last step: {float(last[loss_idx]):.6f}")


if __name__ == "__main__":
    main()

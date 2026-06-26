import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from soap.apps.training.torch_runs import make_dataset, _make_mlp, _safe_float, _grad_norm
from soap.apps.training.representation import activation_covariance, effective_rank_from_covariance, representation_variance, collapse_score

HEADER = ["timestamp", "loss", "val_loss", "grad_norm", "learning_rate", "train_val_gap", "output_variance", "effective_rank", "representation_variance", "collapse_score"]


def hidden_activations(model, X):
    """返回最后一个隐藏层激活（最后 Linear 前），shape (n, last_hidden) tensor。"""
    layers = list(model.children())
    h = X
    for layer in layers[:-1]:
        h = layer(h)
    return h


def run_normal_repr_enhanced(out_csv, steps=300, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [32, 16])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    last_loss = last_val = last_gn = last_var = last_eff = last_repv = last_col = 1.0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
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
                out_var = float(torch.var(val_pred).item())
                hidden = hidden_activations(model, X_val)
                cov = activation_covariance(hidden)
                eff_rank = effective_rank_from_covariance(cov)
                rep_var = representation_variance(hidden)
                collapse = collapse_score(hidden)
            loss_val = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            loss_val = _safe_float(loss_val, last_loss)
            val_loss = _safe_float(val_loss, last_val)
            gn = _safe_float(gn, last_gn)
            out_var = _safe_float(out_var, last_var)
            eff_rank = _safe_float(eff_rank, last_eff)
            rep_var = _safe_float(rep_var, last_repv)
            collapse = _safe_float(collapse, last_col)
            train_val_gap = val_loss - loss_val
            last_loss = loss_val
            last_val = val_loss
            last_gn = gn
            last_var = out_var
            last_eff = eff_rank
            last_repv = rep_var
            last_col = collapse
            writer.writerow([f"step_{step}", loss_val, val_loss, gn, lr, train_val_gap, out_var, eff_rank, rep_var, collapse])
    return out_csv, loss_val, eff_rank, collapse


def run_divergence_repr_enhanced(out_csv, steps=300, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [32, 16])
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    batch_size = 16
    last_loss = last_val = last_gn = last_var = last_eff = last_repv = last_col = 1.0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
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
                out_var = float(torch.var(val_pred).item())
                hidden = hidden_activations(model, X_val)
                cov = activation_covariance(hidden)
                eff_rank = effective_rank_from_covariance(cov)
                rep_var = representation_variance(hidden)
                collapse = collapse_score(hidden)
            loss_val = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            loss_val = _safe_float(loss_val, last_loss)
            val_loss = _safe_float(val_loss, last_val)
            gn = _safe_float(gn, last_gn)
            out_var = _safe_float(out_var, last_var)
            eff_rank = _safe_float(eff_rank, last_eff)
            rep_var = _safe_float(rep_var, last_repv)
            collapse = _safe_float(collapse, last_col)
            train_val_gap = val_loss - loss_val
            last_loss = loss_val
            last_val = val_loss
            last_gn = gn
            last_var = out_var
            last_eff = eff_rank
            last_repv = rep_var
            last_col = collapse
            writer.writerow([f"step_{step}", loss_val, val_loss, gn, lr, train_val_gap, out_var, eff_rank, rep_var, collapse])
    return out_csv, loss_val, eff_rank, collapse


def run_overfit_repr_enhanced(out_csv, steps=400, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    X_train, y_train, X_val, y_val = make_dataset(30, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [128, 64])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    last_loss = last_val = last_gn = last_var = last_eff = last_repv = last_col = 1.0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
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
                out_var = float(torch.var(val_pred).item())
                hidden = hidden_activations(model, X_val)
                cov = activation_covariance(hidden)
                eff_rank = effective_rank_from_covariance(cov)
                rep_var = representation_variance(hidden)
                collapse = collapse_score(hidden)
            loss_val = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            loss_val = _safe_float(loss_val, last_loss)
            val_loss = _safe_float(val_loss, last_val)
            gn = _safe_float(gn, last_gn)
            out_var = _safe_float(out_var, last_var)
            eff_rank = _safe_float(eff_rank, last_eff)
            rep_var = _safe_float(rep_var, last_repv)
            collapse = _safe_float(collapse, last_col)
            train_val_gap = val_loss - loss_val
            last_loss = loss_val
            last_val = val_loss
            last_gn = gn
            last_var = out_var
            last_eff = eff_rank
            last_repv = rep_var
            last_col = collapse
            writer.writerow([f"step_{step}", loss_val, val_loss, gn, lr, train_val_gap, out_var, eff_rank, rep_var, collapse])
    return out_csv, loss_val, eff_rank, collapse


def run_mode_collapse_repr_enhanced(out_csv, steps=300, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [64, 32])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5.0)
    last_loss = last_val = last_gn = last_var = last_eff = last_repv = last_col = 1.0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
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
                out_var = float(torch.var(val_pred).item())
                hidden = hidden_activations(model, X_val)
                cov = activation_covariance(hidden)
                eff_rank = effective_rank_from_covariance(cov)
                rep_var = representation_variance(hidden)
                collapse = collapse_score(hidden)
            loss_val = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            loss_val = _safe_float(loss_val, last_loss)
            val_loss = _safe_float(val_loss, last_val)
            gn = _safe_float(gn, last_gn)
            out_var = _safe_float(out_var, last_var)
            eff_rank = _safe_float(eff_rank, last_eff)
            rep_var = _safe_float(rep_var, last_repv)
            collapse = _safe_float(collapse, last_col)
            train_val_gap = val_loss - loss_val
            last_loss = loss_val
            last_val = val_loss
            last_gn = gn
            last_var = out_var
            last_eff = eff_rank
            last_repv = rep_var
            last_col = collapse
            writer.writerow([f"step_{step}", loss_val, val_loss, gn, lr, train_val_gap, out_var, eff_rank, rep_var, collapse])
    return out_csv, loss_val, eff_rank, collapse


def main():
    examples_dir = Path(__file__).resolve().parents[3] / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    print("Running normal repr enhanced training...")
    csv_normal, loss_n, eff_rank_n, collapse_n = run_normal_repr_enhanced(examples_dir / "torch_training_normal_repr_enhanced.csv")
    print(f"Saved: {csv_normal}")
    print(f"Normal: last_loss={loss_n:.4f}, last_eff_rank={eff_rank_n:.4f}, last_collapse={collapse_n:.4f}")
    print("Running divergence repr enhanced training...")
    csv_divergence, loss_d, eff_rank_d, collapse_d = run_divergence_repr_enhanced(examples_dir / "torch_training_divergence_repr_enhanced.csv")
    print(f"Saved: {csv_divergence}")
    print(f"Divergence: last_loss={loss_d:.4f}, last_eff_rank={eff_rank_d:.4f}, last_collapse={collapse_d:.4f}")
    print("Running overfit repr enhanced training...")
    csv_overfit, loss_o, eff_rank_o, collapse_o = run_overfit_repr_enhanced(examples_dir / "torch_training_overfit_repr_enhanced.csv")
    print(f"Saved: {csv_overfit}")
    print(f"Overfit: last_loss={loss_o:.4f}, last_eff_rank={eff_rank_o:.4f}, last_collapse={collapse_o:.4f}")
    print("Running mode_collapse repr enhanced training...")
    csv_mode_collapse, loss_m, eff_rank_m, collapse_m = run_mode_collapse_repr_enhanced(examples_dir / "torch_training_mode_collapse_repr_enhanced.csv")
    print(f"Saved: {csv_mode_collapse}")
    print(f"Mode Collapse: last_loss={loss_m:.4f}, last_eff_rank={eff_rank_m:.4f}, last_collapse={collapse_m:.4f}")


if __name__ == "__main__":
    main()

# soap/apps/training/torch_multilayer_repr_runs.py
"""Multi-layer representation analysis training runs with comprehensive metrics."""

import csv
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from soap.apps.training.torch_runs import make_dataset, _make_mlp, _safe_float, _grad_norm
from soap.apps.training.representation import activation_covariance, effective_rank_from_covariance, representation_variance, collapse_score

HEADER = ["timestamp", "layer_name", "loss", "val_loss", "grad_norm", "learning_rate", "effective_rank", "representation_variance", "collapse_score"]


def multilayer_activations(model, X):
    """Return [(layer_name, activation_tensor)] for each ReLU output layer."""
    layers = list(model.children())
    h = X
    acts = []
    idx = 0
    for layer in layers[:-1]:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            acts.append((f"hidden_{idx}", h))
            idx += 1
    return acts


def run_normal_multilayer(out_csv, steps=300, seed=42):
    """Run normal training with multi-layer representation tracking."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [32, 16])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    last_loss = float('inf')
    last_val = float('inf')
    last_gn = float('inf')
    last_eff = {}
    last_repv = {}
    last_col = {}
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        
        for step in range(steps):
            model.train()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            gn = _grad_norm(model)
            optimizer.step()
            optimizer.zero_grad()
            
            loss_val = _safe_float(loss.item(), last_loss)
            last_loss = loss_val
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = _safe_float(criterion(val_pred, y_val).item(), last_val)
                last_val = val_loss
                
                acts = multilayer_activations(model, X_val)
                for layer_name, hidden in acts:
                    cov = activation_covariance(hidden)
                    eff_rank = _safe_float(effective_rank_from_covariance(cov), 
                                          last_eff.get(layer_name, float('inf')))
                    last_eff[layer_name] = eff_rank
                    
                    rep_var = _safe_float(representation_variance(hidden), 
                                         last_repv.get(layer_name, float('inf')))
                    last_repv[layer_name] = rep_var
                    
                    collapse = _safe_float(collapse_score(hidden), 
                                          last_col.get(layer_name, float('inf')))
                    last_col[layer_name] = collapse
                    
                    gn_val = _safe_float(gn, last_gn)
                    last_gn = gn_val
                    
                    writer.writerow([
                        f"step_{step}",
                        layer_name,
                        loss_val,
                        val_loss,
                        gn_val,
                        1e-3,
                        eff_rank,
                        rep_var,
                        collapse
                    ])


def run_divergence_multilayer(out_csv, steps=300, seed=42):
    """Run divergence training with mini-batch SGD and multi-layer representation tracking."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [32, 16])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.MSELoss()
    batch_size = 16
    
    last_loss = float('inf')
    last_val = float('inf')
    last_gn = float('inf')
    last_eff = {}
    last_repv = {}
    last_col = {}
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        
        for step in range(steps):
            model.train()
            idx = torch.randperm(X_train.size(0))[:batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            gn = _grad_norm(model)
            optimizer.step()
            optimizer.zero_grad()
            
            loss_val = _safe_float(loss.item(), last_loss)
            last_loss = loss_val
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = _safe_float(criterion(val_pred, y_val).item(), last_val)
                last_val = val_loss
                
                acts = multilayer_activations(model, X_val)
                for layer_name, hidden in acts:
                    cov = activation_covariance(hidden)
                    eff_rank = _safe_float(effective_rank_from_covariance(cov), 
                                          last_eff.get(layer_name, float('inf')))
                    last_eff[layer_name] = eff_rank
                    
                    rep_var = _safe_float(representation_variance(hidden), 
                                         last_repv.get(layer_name, float('inf')))
                    last_repv[layer_name] = rep_var
                    
                    collapse = _safe_float(collapse_score(hidden), 
                                          last_col.get(layer_name, float('inf')))
                    last_col[layer_name] = collapse
                    
                    gn_val = _safe_float(gn, last_gn)
                    last_gn = gn_val
                    
                    writer.writerow([
                        f"step_{step}",
                        layer_name,
                        loss_val,
                        val_loss,
                        gn_val,
                        0.5,
                        eff_rank,
                        rep_var,
                        collapse
                    ])


def run_overfit_multilayer(out_csv, steps=400, seed=42):
    """Run overfitting training with multi-layer representation tracking."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    X_train, y_train, X_val, y_val = make_dataset(30, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [128, 64])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.MSELoss()
    
    last_loss = float('inf')
    last_val = float('inf')
    last_gn = float('inf')
    last_eff = {}
    last_repv = {}
    last_col = {}
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        
        for step in range(steps):
            model.train()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            gn = _grad_norm(model)
            optimizer.step()
            optimizer.zero_grad()
            
            loss_val = _safe_float(loss.item(), last_loss)
            last_loss = loss_val
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = _safe_float(criterion(val_pred, y_val).item(), last_val)
                last_val = val_loss
                
                acts = multilayer_activations(model, X_val)
                for layer_name, hidden in acts:
                    cov = activation_covariance(hidden)
                    eff_rank = _safe_float(effective_rank_from_covariance(cov), 
                                          last_eff.get(layer_name, float('inf')))
                    last_eff[layer_name] = eff_rank
                    
                    rep_var = _safe_float(representation_variance(hidden), 
                                         last_repv.get(layer_name, float('inf')))
                    last_repv[layer_name] = rep_var
                    
                    collapse = _safe_float(collapse_score(hidden), 
                                          last_col.get(layer_name, float('inf')))
                    last_col[layer_name] = collapse
                    
                    gn_val = _safe_float(gn, last_gn)
                    last_gn = gn_val
                    
                    writer.writerow([
                        f"step_{step}",
                        layer_name,
                        loss_val,
                        val_loss,
                        gn_val,
                        1e-3,
                        eff_rank,
                        rep_var,
                        collapse
                    ])


def run_mode_collapse_multilayer(out_csv, steps=300, seed=42):
    """Run mode collapse training with multi-layer representation tracking."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    X_train, y_train, X_val, y_val = make_dataset(200, n_val=200, seed=seed)
    model = _make_mlp(X_train.shape[1], [64, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5.0)
    criterion = nn.MSELoss()
    
    last_loss = float('inf')
    last_val = float('inf')
    last_gn = float('inf')
    last_eff = {}
    last_repv = {}
    last_col = {}
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        
        for step in range(steps):
            model.train()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            gn = _grad_norm(model)
            optimizer.step()
            optimizer.zero_grad()
            
            loss_val = _safe_float(loss.item(), last_loss)
            last_loss = loss_val
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = _safe_float(criterion(val_pred, y_val).item(), last_val)
                last_val = val_loss
                
                acts = multilayer_activations(model, X_val)
                for layer_name, hidden in acts:
                    cov = activation_covariance(hidden)
                    eff_rank = _safe_float(effective_rank_from_covariance(cov), 
                                          last_eff.get(layer_name, float('inf')))
                    last_eff[layer_name] = eff_rank
                    
                    rep_var = _safe_float(representation_variance(hidden), 
                                         last_repv.get(layer_name, float('inf')))
                    last_repv[layer_name] = rep_var
                    
                    collapse = _safe_float(collapse_score(hidden), 
                                          last_col.get(layer_name, float('inf')))
                    last_col[layer_name] = collapse
                    
                    gn_val = _safe_float(gn, last_gn)
                    last_gn = gn_val
                    
                    writer.writerow([
                        f"step_{step}",
                        layer_name,
                        loss_val,
                        val_loss,
                        gn_val,
                        1e-3,
                        eff_rank,
                        rep_var,
                        collapse
                    ])


def main():
    """Run all multilayer representation training experiments."""
    examples_dir = Path(__file__).resolve().parents[3] / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Run normal training
    normal_csv = examples_dir / "torch_training_normal_multilayer_repr.csv"
    run_normal_multilayer(normal_csv)
    print(f"Generated: {normal_csv}")
    
    # Run divergence training
    divergence_csv = examples_dir / "torch_training_divergence_multilayer_repr.csv"
    run_divergence_multilayer(divergence_csv)
    print(f"Generated: {divergence_csv}")
    
    # Run overfit training
    overfit_csv = examples_dir / "torch_training_overfit_multilayer_repr.csv"
    run_overfit_multilayer(overfit_csv)
    print(f"Generated: {overfit_csv}")
    
    # Run mode collapse training
    mode_collapse_csv = examples_dir / "torch_training_mode_collapse_multilayer_repr.csv"
    run_mode_collapse_multilayer(mode_collapse_csv)
    print(f"Generated: {mode_collapse_csv}")
    
    # Print final effective rank and collapse score for each layer in each experiment
    experiments = [
        ("normal", normal_csv),
        ("divergence", divergence_csv),
        ("overfit", overfit_csv),
        ("mode_collapse", mode_collapse_csv)
    ]
    
    for name, csv_path in experiments:
        print(f"\n{name} experiment final state:")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            last_rows = {}
            for row in reader:
                layer_name = row['layer_name']
                last_rows[layer_name] = row
            
            for layer_name, row in last_rows.items():
                print(f"  {layer_name}: effective_rank={row['effective_rank']}, collapse_score={row['collapse_score']}")


if __name__ == "__main__":
    main()

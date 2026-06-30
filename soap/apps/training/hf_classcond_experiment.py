import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast

from soap.apps.training.hf_repr_experiment import (
    DistilBertClassifier, make_structured_dataset, tokenize_texts,
    alpha_schedule, compute_probe_first_pc)
from soap.apps.training.classcond_metrics import (
    nc1_ratio, centroid_effective_rank, fisher_separation, etf_deviation)


def _safe_float(val, default=0.0):
    """安全转换浮点数，防止NaN/Inf"""
    if val is None:
        return default
    f = float(val)
    if np.isnan(f) or np.isinf(f):
        return default
    return f


def compute_classcond(model, probe_ids, probe_mask, probe_labels, last_values,
                      alpha, proj_active, unit_vec):
    """
    计算每层的类条件几何指标（基于独立 probe split）。

    返回: [(layer_i, nc1, crk, fish, etf), ...] 共 7 层（embedding + 6 transformer）。
    accuracy 不在此计算（见 compute_val_accuracy，基于独立 val split）。
    """
    model.eval()
    with torch.no_grad():
        _, hs = model(probe_ids, probe_mask, alpha=alpha, proj_active=proj_active, unit_vec=unit_vec)

    # 基于 attention_mask 的均值池化每层隐藏状态
    mask = probe_mask.unsqueeze(-1).float()  # (B, S, 1)
    results = []
    num_layers = 7  # DistilBERT: embedding + 6 transformer layers
    for i in range(num_layers):
        pooled = ((hs[i] * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()  # (B, 768)
        X = pooled
        y = probe_labels.cpu().numpy()

        nc1 = _safe_float(nc1_ratio(X, y, 5), last_values.get(f"nc1_{i}", 0.0))
        crk = _safe_float(centroid_effective_rank(X, y, 5), last_values.get(f"crk_{i}", 0.0))
        fish = _safe_float(fisher_separation(X, y, 5), last_values.get(f"fish_{i}", 0.0))
        etf = _safe_float(etf_deviation(X, y, 5), last_values.get(f"etf_{i}", 0.0))

        last_values[f"nc1_{i}"] = nc1
        last_values[f"crk_{i}"] = crk
        last_values[f"fish_{i}"] = fish
        last_values[f"etf_{i}"] = etf

        results.append((i, nc1, crk, fish, etf))

    return results


def compute_val_accuracy(model, val_ids, val_mask, val_labels):
    """
    在独立 validation split 上计算分类准确率。

    采用 normal 前向（alpha=0, proj_active=False）：accuracy 衡量模型标准推理
    功能保持情况，与 geometry 指标（可按 condition 前向提取干预后表示）解耦。
    判据：良性 NC → accuracy 保持；病态 collapse → accuracy 同步恶化。
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(val_ids, val_mask, alpha=0.0, proj_active=False, unit_vec=None)
    return (logits.argmax(-1) == val_labels).float().mean().item()


def run_experiment(seed, condition, steps, batch_size, record_every, probe_size, device, output_path):
    """运行类条件实验"""

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 加载数据集：6 元组解包，train/val/probe 三池严格分离
    texts_train, labels_train, texts_val, labels_val, texts_probe, labels_probe = \
        make_structured_dataset(data_seed=0)

    # tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased", local_files_only=True
    )

    # 模型（num_classes 从 labels_train 推导；DistilBertClassifier 参数名为 num_classes）
    num_classes = max(labels_train) + 1
    model = DistilBertClassifier(num_classes=num_classes)
    model.to(device)

    # 冻结前 3 层（含 embeddings），复用 v0.7.4/5 的 freeze_front
    model.freeze_front(n=3)

    # probe split（独立，用于 class-conditional geometry 指标；不参与训练）
    n_probe = min(probe_size, len(texts_probe))
    probe_ids, probe_mask = tokenize_texts(texts_probe[:n_probe], tokenizer, max_length=128)
    probe_ids = probe_ids.to(device)
    probe_mask = probe_mask.to(device)
    probe_labels = torch.tensor(labels_probe[:n_probe], dtype=torch.long).to(device)

    # validation split（独立，用于 validation_accuracy；不参与训练）
    val_ids, val_mask = tokenize_texts(texts_val, tokenizer, max_length=128)
    val_ids = val_ids.to(device)
    val_mask = val_mask.to(device)
    val_labels = torch.tensor(labels_val, dtype=torch.long).to(device)

    # rank-1 投影方向：初始 probe layer_4 第一主成分（全 run 固定）
    unit_vec = compute_probe_first_pc(model, probe_ids, probe_mask, device)

    # 优化器（仅未冻结参数）
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5, weight_decay=0,
    )

    criterion = nn.CrossEntropyLoss()

    # 训练池：仅 train split（probe/val 不入训练索引）
    n_train = len(texts_train)

    # CSV 写入（validation_accuracy 为独立列，同 checkpoint 7 层行共享；无 accuracy 伪层）
    csv_header = ["seed", "condition", "timestamp", "layer_name", "nc1_ratio",
                  "centroid_effective_rank", "fisher_ratio", "etf_deviation",
                  "validation_accuracy"]
    csv_file = open(output_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    # last_values 防 NaN 兜底缓存
    last_values = {}

    # batch 采样生成器（种子固定）
    g = torch.Generator()
    g.manual_seed(seed)

    print(f"开始实验: seed={seed}, condition={condition}, steps={steps}, batch_size={batch_size}")
    print(f"train={n_train}, val={len(texts_val)}, probe={n_probe}, 记录间隔={record_every}")

    for step in range(1, steps + 1):
        model.train()

        # 随机采样一个 batch（仅来自 train split）
        batch_indices = torch.randint(
            low=0, high=n_train, size=(batch_size,), generator=g
        ).tolist()
        batch_texts = [texts_train[i] for i in batch_indices]
        batch_labels_list = [labels_train[i] for i in batch_indices]

        batch_ids, batch_mask = tokenize_texts(batch_texts, tokenizer, max_length=128)
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = torch.tensor(batch_labels_list, dtype=torch.long).to(device)

        optimizer.zero_grad()

        if condition == "condition_b":
            # 条件B：alpha 调度 + rank-1 干预
            alpha = alpha_schedule(step, steps)
            logits, _ = model(batch_ids, batch_mask, alpha=alpha, proj_active=True, unit_vec=unit_vec)
        else:
            # 正常条件：alpha=0，无干预
            logits, _ = model(batch_ids, batch_mask, alpha=0.0, proj_active=False, unit_vec=unit_vec)

        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        # 定期记录类条件指标
        if step % record_every == 0:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            if condition == "condition_b":
                alpha = alpha_schedule(step, steps)
                results = compute_classcond(
                    model, probe_ids, probe_mask, probe_labels, last_values,
                    alpha=alpha, proj_active=True, unit_vec=unit_vec
                )
            else:
                results = compute_classcond(
                    model, probe_ids, probe_mask, probe_labels, last_values,
                    alpha=0.0, proj_active=False, unit_vec=unit_vec
                )

            # validation accuracy（独立 val split，normal 前向，强校验 [0,1]）
            val_acc = _safe_float(
                compute_val_accuracy(model, val_ids, val_mask, val_labels),
                last_values.get("val_acc", 0.0),
            )
            assert 0.0 <= val_acc <= 1.0, f"validation_accuracy 越界 [0,1]: {val_acc}"
            last_values["val_acc"] = val_acc

            # 写入 7 层行（同 checkpoint 共享同一 validation_accuracy）
            for layer_i, nc1_val, crk_val, fish_val, etf_val in results:
                csv_writer.writerow([
                    seed, condition, timestamp, f"layer_{layer_i}",
                    nc1_val, crk_val, fish_val, etf_val, val_acc
                ])

            csv_file.flush()
            print(f"  Step {step}/{steps} 已记录 ({condition}) val_acc={val_acc:.4f}")

    csv_file.close()
    print(f"实验完成，结果已保存至 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="类条件实验 (HF DistilBERT)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--condition", type=str, default="normal",
                        choices=["normal", "condition_b"], help="实验条件")
    parser.add_argument("--steps", type=int, default=120, help="训练总步数")
    parser.add_argument("--batch-size", type=int, default=16, help="批大小")
    parser.add_argument("--record-every", type=int, default=3, help="记录间隔步数")
    parser.add_argument("--probe-size", type=int, default=128, help="探针集大小")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/mps/cuda/cpu)")
    parser.add_argument("--output", type=str, required=True, help="输出 CSV 路径")

    args = parser.parse_args()

    # 自动检测设备
    if args.device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"使用设备: {device}")

    run_experiment(
        seed=args.seed,
        condition=args.condition,
        steps=args.steps,
        batch_size=args.batch_size,
        record_every=args.record_every,
        probe_size=args.probe_size,
        device=device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

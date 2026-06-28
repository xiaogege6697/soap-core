"""
hf_repr_experiment.py — 第一阶段：数据生成与分类模型

本模块实现：
1. make_structured_dataset() 生成 5 类结构化英文短句数据集
2. DistilBertClassifier 基于 DistilBert 的文本分类器
后续阶段将追加几何特征记录与训练循环。
"""

import random
import argparse
from itertools import product
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

import csv
from soap.apps.training.representation import activation_covariance, effective_rank_from_covariance, representation_variance, collapse_score
from soap.apps.training.torch_runs import _safe_float


# ---------------------------------------------------------------------------
# 数据生成
# ---------------------------------------------------------------------------

def make_structured_dataset(data_seed=0):
    """
    生成 5 类结构化英文短句数据集。

    5 类主题：旅行(0) / 饮食(1) / 天气(2) / 运动(3) / 科技(4)
    每类用固定句式模板加词槽词表，类与类之间的实词词汇不重叠，
    每类组合数 >= 350。

    返回:
        texts_train, labels_train,
        texts_val,   labels_val,
        texts_probe, labels_probe
        labels 为 0-4 的整数列表。
    """
    rng = random.Random(data_seed)
    np.random.seed(data_seed)

    # ---- 各类词槽（类间实词不重叠）----------------------------------------

    # 旅行 (label=0)
    travel_places = [
        "Paris", "London", "Tokyo", "Rome", "Berlin", "Sydney", "Moscow",
        "Cairo", "Mumbai", "Bangkok", "Istanbul", "Lisbon", "Prague",
        "Dublin", "Vienna", "Oslo", "Lima", "Havana", "Seoul", "Hanoi",
        "Athens", "Zurich", "Nairobi", "Bogota", "Jakarta",
    ]
    travel_freq = [
        "always", "often", "sometimes", "rarely", "occasionally",
        "frequently", "regularly", "yearly", "annually", "twice",
    ]
    travel_season = [
        "spring", "summer", "autumn", "winter", "monsoon",
        "harvest", "festive", "carnival", "pilgrimage", "foliage",
    ]

    # 饮食 (label=1)
    food_adj = [
        "delicious", "tasty", "savory", "spicy", "sweet", "sour", "crispy",
        "tender", "creamy", "rich", "mild", "fresh", "salty", "bitter",
        "bland", "roasted", "grilled", "baked", "fried", "smoked",
        "stuffed", "glazed", "poached", "steamed", "marinated",
    ]
    food_dish = [
        "pasta", "sushi", "steak", "curry", "salad", "soup", "pizza",
        "ramen", "tacos", "dumplings", "falafel", "risotto", "lasagna",
        "paella", "kebab", "tempura", "ceviche", "fondue", "gnocchi",
        "bruschetta",
    ]

    # 天气 (label=2)
    weather_adj = [
        "sunny", "rainy", "cloudy", "windy", "foggy", "snowy", "humid",
        "dry", "warm", "cold", "stormy", "freezing", "scorching", "breezy",
        "overcast", "drizzly", "blustery", "sweltering", "frigid", "severe",
        "arid", "muggy", "chilly", "gusty", "hazy",
    ]
    weather_region = [
        "coastline", "highlands", "lowlands", "valley", "plateau", "basin",
        "prairie", "tundra", "desert", "jungle", "steppe", "marshland",
        "foothills", "moorland", "grassland", "wetland", "farmland",
        "woodland", "hilltop", "canyon",
    ]

    # 运动 (label=3)
    sport_type = [
        "tennis", "golf", "boxing", "wrestling", "fencing", "archery",
        "cycling", "rowing", "skiing", "skating", "diving", "surfing",
        "climbing", "sailing", "triathlon", "marathon", "decathlon",
        "pentathlon", "biathlon", "badminton", "squash", "lacrosse",
        "handball", "softball", "waterpolo",
    ]
    sport_venue = [
        "arena", "stadium", "court", "rink", "complex", "park", "oval",
        "dome", "pavilion", "coliseum", "amphitheater", "grounds",
        "clubhouse", "track", "range", "circuit", "dojo", "ring",
        "gymnasium", "velodrome",
    ]

    # 科技 (label=4)  26×20 = 520 种组合（>= 425：train320+val80+probe25）
    tech_device = [
        "sensor", "router", "scanner", "printer", "modem", "processor",
        "chipset", "transceiver", "actuator", "oscillator", "antenna",
        "diode", "capacitor", "amplifier", "encoder", "decoder",
        "comparator", "detector", "gyroscope", "motherboard",
        "transmitter", "monitor", "battery", "display", "speaker",
        "microcontroller",
    ]
    tech_tool = [
        "debugger", "simulator", "profiler", "emulator", "compiler",
        "validator", "verifier", "analyzer", "synthesizer", "optimizer",
        "calibrator", "serializer", "parser", "generator", "aggregator",
        "dispatcher", "controller", "configurator", "normalizer", "depurator",
    ]

    # ---- 句式模板与对应词槽 ------------------------------------------------
    # (模板字符串, 词槽字典)
    # 词槽字典的 key 用单字母 a/b/c，模板中对应 {a}/{b}/{c}
    class_specs = [
        # label=0 旅行  25×10×10 = 2500 种组合
        ("the traveler visited {a} {b} during the {c} holiday",
         {"a": travel_places, "b": travel_freq, "c": travel_season}),
        # label=1 饮食  25×20 = 500 种组合
        ("the chef served {a} {b} to the guests at dinner",
         {"a": food_adj, "b": food_dish}),
        # label=2 天气  25×20 = 500 种组合
        ("the forecast predicted {a} weather across the {b}",
         {"a": weather_adj, "b": weather_region}),
        # label=3 运动  25×20 = 500 种组合
        ("the athlete won the {a} event at the {b}",
         {"a": sport_type, "b": sport_venue}),
        # label=4 科技  20×20 = 400 种组合
        ("the engineer tested the {a} with the {b}",
         {"a": tech_device, "b": tech_tool}),
    ]

    # ---- 各 split 规模 ----------------------------------------------------
    n_classes = 5
    n_train_per_class = 320          # train 共 1600
    n_val_per_class = 80             # val 共 400
    n_probe_total = 128              # probe 共 128
    n_probe_base = n_probe_total // n_classes   # 25
    n_probe_extra = n_probe_total % n_classes   # 3（分给前 3 类）

    texts_train, labels_train = [], []
    texts_val,   labels_val   = [], []
    texts_probe, labels_probe = [], []

    for label, (template, slots) in enumerate(class_specs):
        # 固定 key 排序，保证确定性
        slot_names = sorted(slots.keys())
        slot_lists = [slots[k] for k in slot_names]

        # 生成该类所有可能的文本组合
        all_texts = [
            template.format(**dict(zip(slot_names, combo)))
            for combo in product(*slot_lists)
        ]

        # 用固定种子打乱
        rng.shuffle(all_texts)

        # 计算该类 probe 数量
        n_probe = n_probe_base + (1 if label < n_probe_extra else 0)
        n_needed = n_train_per_class + n_val_per_class + n_probe
        assert len(all_texts) >= n_needed, (
            f"类别 {label} 组合数 {len(all_texts)} < 需要的 {n_needed}"
        )

        idx = 0
        # train
        texts_train.extend(all_texts[idx: idx + n_train_per_class])
        labels_train.extend([label] * n_train_per_class)
        idx += n_train_per_class
        # val
        texts_val.extend(all_texts[idx: idx + n_val_per_class])
        labels_val.extend([label] * n_val_per_class)
        idx += n_val_per_class
        # probe
        texts_probe.extend(all_texts[idx: idx + n_probe])
        labels_probe.extend([label] * n_probe)

    return texts_train, labels_train, texts_val, labels_val, texts_probe, labels_probe


# ---------------------------------------------------------------------------
# 分类模型
# ---------------------------------------------------------------------------

class DistilBertClassifier(nn.Module):
    """基于 DistilBert 的 5 类文本分类器。"""

    def __init__(self, num_classes=5, dropout=0.1):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", output_hidden_states=True, local_files_only=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )
        # controlled rank-1 projection 干预状态（condition_b 用）
        # 通过 forward hook 修改 layer[3](=block3=hidden_states[4]) 的输出，
        # 使干预作用于送入后续 block 的完整 hidden states，可观察 layer_4→5→6 传播。
        self._proj = {"active": False, "alpha": 0.0, "mask": None, "unit_vec": None}
        self._proj_handle = self.backbone.transformer.layer[3].register_forward_hook(
            self._proj_hook
        )

    def _proj_hook(self, module, inp, out):
        """
        在 block3(layer[3]=hidden_states[4]) 输出处施加 pooled-space rank-1 投影干预。
        在 sample pooled 空间构造 rank-1 目标，再把位移写回完整 hidden states，
        使 masked_mean(modified) == target —— 与 eff_rank 同空间，且能传入后续 block。
        仅 condition_b 且 alpha>0 时生效。
        """
        st = self._proj
        if not st["active"] or st["alpha"] <= 0.0 or st["unit_vec"] is None:
            return None
        hidden = out[0] if isinstance(out, (tuple, list)) else out  # (B, L, 768)
        mask = st["mask"]
        u = st["unit_vec"]                          # (768,) 固定第一主方向
        alpha = st["alpha"]
        m = mask.unsqueeze(-1).float()              # (B, L, 1)
        denom = m.sum(dim=1).clamp(min=1.0)         # (B, 1)
        # 1) 每样本 masked pooled 表示
        pooled = (hidden * m).sum(dim=1) / denom    # (B, 768)
        # 2) 跨样本全局中心（关键：非每样本 token mean，避免保留样本间满秩差异）
        global_center = pooled.mean(dim=0, keepdim=True)   # (1, 768)
        centered = pooled - global_center           # (B, 768)
        # 3) rank-1 pooled 目标：能量集中到 u 方向
        proj_scalar = (centered * u).sum(dim=-1, keepdim=True)   # (B, 1)
        target = global_center + (1.0 - alpha) * centered + alpha * (proj_scalar * u)  # (B, 768)
        # 4) 写回完整 hidden：correction 对该样本所有 token 相同，仅加到非 pad 位
        correction = (target - pooled).unsqueeze(1)  # (B, 1, 768)
        modified = hidden + correction * m          # (B, L, 768)
        # 5) 运行时断言：masked_mean(modified) == target（数学上应 <1e-5，1e-4 留浮点/MPS 余量）
        check = ((modified * m).sum(dim=1) / denom - target).abs().max().item()
        assert check < 1e-4, f"rank-1 写回不一致: max_abs={check}"
        if isinstance(out, (tuple, list)):
            return (modified,) + tuple(out[1:])
        return modified

    def forward(self, input_ids, attention_mask, alpha=0.0, proj_active=False, unit_vec=None):
        """
        前向传播。

        参数:
            alpha:       rank-1 投影混合系数（0=原始，→0.98 几乎全 rank-1）
            proj_active: 是否激活 rank-1 投影干预（仅 condition_b 为 True）
            unit_vec:    固定单位投影向量 (768,)，每 seed 确定性生成、全 run 不更新

        返回:
            logits:        (B, num_classes) 分类 logits
            hidden_states: 含 7 个张量的元组（embedding + 6 层），每个 (B, seq_len, 768)
        """
        # 更新投影干预状态（供 _proj_hook 读取）
        self._proj.update(
            active=bool(proj_active and alpha > 0.0),
            alpha=float(alpha),
            mask=attention_mask,
            unit_vec=unit_vec,
        )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: (embedding_out, layer0_out, ..., layer5_out)
        hidden_states = outputs.hidden_states  # 7 个张量的元组

        # 取最后一层隐藏状态
        last = hidden_states[-1]  # (B, seq_len, 768)

        # 基于 attention_mask 的加权平均池化（忽略 pad 位置）
        mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, 768)

        # 分类头
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits, hidden_states

    def freeze_front(self, n=3):
        """
        冻结前 n 个 Transformer block（含 embeddings）。

        冻结范围：
        - backbone.embeddings 的全部参数
        - backbone.transformer.layer[0] 到 layer[n-1] 的全部参数
        """
        # 冻结 embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad_(False)
        # 冻结前 n 个 Transformer 层
        for i in range(n):
            for param in self.backbone.transformer.layer[i].parameters():
                param.requires_grad_(False)


# ---------------------------------------------------------------------------
# 第二阶段：几何特征记录
# ---------------------------------------------------------------------------

METRICS_HEADER = [
    "seed", "condition", "timestamp", "layer_name",
    "loss", "val_loss", "grad_norm", "learning_rate",
    "effective_rank", "representation_variance", "collapse_score",
]
# CSV 表头常量，用于几何特征记录文件


def open_metrics_csv(path):
    """
    以写方式打开 CSV 文件并写入表头。

    参数:
        path: CSV 文件路径

    返回:
        (file_obj, writer) 二元组，file_obj 需由调用方负责关闭。
    """
    file_obj = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_obj)
    writer.writerow(METRICS_HEADER)
    return file_obj, writer


def write_metrics_row(writer, seed, condition, timestamp, layer_name,
                      loss, val_loss, grad_norm, learning_rate,
                      eff_rank, rep_var, collapse_score):
    """
    按 METRICS_HEADER 顺序写入一行几何特征记录。

    参数:
        writer: csv.writer 对象
        seed: 随机种子
        condition: 实验条件标识
        timestamp: 时间戳字符串
        layer_name: 层名称，如 "layer_0"
        loss: 当前训练损失
        val_loss: 当前验证损失
        grad_norm: 当前梯度范数
        learning_rate: 当前学习率
        eff_rank: 有效秩
        rep_var: 表征方差
        collapse_score: 坍塌分数
    """
    writer.writerow([
        seed, condition, timestamp, layer_name,
        loss, val_loss, grad_norm, learning_rate,
        eff_rank, rep_var, collapse_score,
    ])


def tokenize_texts(texts, tokenizer, max_length=32):
    """
    对文本列表进行 tokenize 并返回 input_ids 与 attention_mask。

    参数:
        texts: 字符串列表
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列长度，默认 32

    返回:
        (input_ids, attention_mask) 两个 tensor，形状均为 (len(texts), max_length)。
    """
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def compute_layer_metrics(model, input_ids, attention_mask, last_values,
                          alpha=0.0, proj_active=False, unit_vec=None):
    """
    在给定的一批固定样本上，对模型每一层隐藏状态计算几何特征。

    参数:
        model: DistilBertClassifier 实例
        input_ids: (B, seq_len) token id tensor
        attention_mask: (B, seq_len) 注意力掩码 tensor
        last_values: dict，跨 checkpoint 传递每个 layer 的上一次有效值
                     用于防 NaN 兜底；首次调用由外部传入空 dict {}

    返回:
        list，元素为 (layer_name, eff_rank, rep_var, col_score)，
        layer_name 格式为 "layer_0", "layer_1", ... 共 7 层（embedding + 6 层）。
    """
    model.eval()
    with torch.no_grad():
        logits, hidden_states = model(
            input_ids, attention_mask,
            alpha=alpha, proj_active=proj_active, unit_vec=unit_vec,
        )
    # hidden_states: 含 7 个张量的元组（embedding + 6 层），每个 (B, seq_len, 768)

    mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)

    results = []
    for i in range(len(hidden_states)):
        h = hidden_states[i]  # (B, seq_len, 768)
        # 基于 mask 的加权平均池化，转为 numpy
        pooled = ((h * mask).sum(dim=1) / mask.sum(dim=1)).detach().cpu().numpy()  # (B, 768)

        # 计算协方差矩阵
        cov = activation_covariance(pooled)

        # 从 last_values 获取该层上一次有效值，用于 NaN 兜底
        prev = last_values.get(i, {})

        # 计算三项几何特征，NaN 时回退到上次值
        eff_rank = _safe_float(
            effective_rank_from_covariance(cov), prev.get("eff", float("inf"))
        )
        rep_var = _safe_float(
            representation_variance(pooled), prev.get("rep", float("inf"))
        )
        col_score = _safe_float(
            collapse_score(pooled), prev.get("col", float("inf"))
        )

        # 更新兜底值
        last_values[i] = {"eff": eff_rank, "rep": rep_var, "col": col_score}

        layer_name = "layer_" + str(i)
        results.append((layer_name, eff_rank, rep_var, col_score))

    return results


# ---------------------------------------------------------------------------
# 第三阶段：训练循环
# ---------------------------------------------------------------------------

def alpha_schedule(step, steps=120, warmup=20, alpha_max=0.98):
    """
    controlled progressive rank-1 投影的 alpha 调度。

    step <= warmup : alpha=0（无干预，正常训练）
    step >  warmup : alpha 线性增到 alpha_max（step=steps 时达 alpha_max）

    参数提前固定，不根据 detector 输出调节。
    """
    if step <= warmup:
        return 0.0
    return min(alpha_max, alpha_max * (step - warmup) / max(1, (steps - warmup)))


def compute_probe_first_pc(model, probe_input_ids, probe_attention_mask, device, layer_idx=4):
    """
    从初始 probe representation 计算指定层的第一主方向（PCA 最大特征值对应特征向量）。
    只用初始 normal/probe 数据（训练前 forward），detach 后整个 run 固定，
    不使用 condition_b 结果调节。
    """
    probe_input_ids = probe_input_ids.to(device)
    probe_attention_mask = probe_attention_mask.to(device)
    model.eval()
    with torch.no_grad():
        _, hidden_states = model(
            probe_input_ids, probe_attention_mask,
            alpha=0.0, proj_active=False, unit_vec=None,
        )
        h = hidden_states[layer_idx]                      # (N, L, 768)
        m = probe_attention_mask.unsqueeze(-1).float()    # (N, L, 1)
        denom = m.sum(dim=1).clamp(min=1.0)               # (N, 1)
        pooled = (h * m).sum(dim=1) / denom               # (N, 768)
        pooled_np = pooled.cpu().numpy()
        cov = np.cov(pooled_np, rowvar=False)             # (768, 768)
        eigvals, eigvecs = np.linalg.eigh(cov)            # 升序特征值/向量
        u = eigvecs[:, -1]                                # 最大特征值方向
        return torch.tensor(u, dtype=torch.float32, device=device)


def run_experiment(seed, condition, steps, batch_size, record_every,
                   probe_size, device, output_path):
    """
    运行一次训练实验，记录几何特征到 CSV。

    condition_b 使用 controlled progressive rank-1 activation intervention：
    在 layer_4(block3) 输出处施加固定 rank-1 投影（alpha 渐进调度），
    只保留分类损失（无 contraction loss / lambda）。

    参数:
        seed:          随机种子（控制模型初始化、dropout、batch 顺序、unit_vec）
        condition:     实验条件，"normal" 或 "condition_b"
        steps:         总训练步数
        batch_size:    批大小
        record_every:  每隔多少步记录一次几何特征
        probe_size:    probe 集大小
        device:        计算设备
        output_path:   输出 CSV 文件路径
    """
    # ---- 设种子 --------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---- 数据（data_seed=0 固定，与 seed 无关）--------------------------
    texts_train, labels_train, \
        texts_val, labels_val, \
        texts_probe, labels_probe = make_structured_dataset(data_seed=0)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", local_files_only=True)

    train_input_ids, train_attention_mask = tokenize_texts(texts_train, tokenizer)
    val_input_ids, val_attention_mask = tokenize_texts(texts_val, tokenizer)

    n_probe = min(probe_size, len(texts_probe))
    probe_input_ids, probe_attention_mask = tokenize_texts(
        texts_probe[:n_probe], tokenizer
    )

    # 标签转 tensor
    train_labels = torch.tensor(labels_train, dtype=torch.long)
    val_labels = torch.tensor(labels_val, dtype=torch.long)

    # ---- 模型 ----------------------------------------------------------
    model = DistilBertClassifier()
    model.freeze_front(n=3)
    model.to(device)

    # ---- rank-1 投影方向：初始 probe layer_4 pooled 的第一主方向（全 run 固定）---
    unit_vec = compute_probe_first_pc(model, probe_input_ids, probe_attention_mask, device)

    # 移动数据到设备
    train_input_ids = train_input_ids.to(device)
    train_attention_mask = train_attention_mask.to(device)
    train_labels = train_labels.to(device)
    val_input_ids = val_input_ids.to(device)
    val_attention_mask = val_attention_mask.to(device)
    val_labels = val_labels.to(device)
    probe_input_ids = probe_input_ids.to(device)
    probe_attention_mask = probe_attention_mask.to(device)

    # ---- 优化器与损失 --------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5, weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")

    # ---- CSV 输出 ------------------------------------------------------
    file_obj, writer = open_metrics_csv(output_path)
    last_values = {}

    # ---- Batch shuffle 准备 -------------------------------------------
    n_train = len(texts_train)
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(seed)
    perm = torch.randperm(n_train, generator=shuffle_gen)
    perm_idx = 0
    epoch = 0

    # ---- 训练循环 ------------------------------------------------------
    for step in range(steps):
        model.train()

        # 若当前 epoch 索引不够，重排（steps 超过一轮时循环重排）
        if perm_idx + batch_size > n_train:
            epoch += 1
            shuffle_gen.manual_seed(seed + epoch)
            perm = torch.randperm(n_train, generator=shuffle_gen)
            perm_idx = 0

        batch_indices = perm[perm_idx: perm_idx + batch_size]
        perm_idx += batch_size

        batch_input_ids = train_input_ids[batch_indices]
        batch_attention_mask = train_attention_mask[batch_indices]
        batch_labels = train_labels[batch_indices]

        # alpha 调度（参数提前固定，不据 detector 输出调节）
        alpha = alpha_schedule(step, steps=steps)
        proj_active = (condition == "condition_b")

        # 前向传播（condition_b 且 alpha>0 时激活 rank-1 投影干预）
        logits, hidden_states = model(
            batch_input_ids, batch_attention_mask,
            alpha=alpha, proj_active=proj_active, unit_vec=unit_vec,
        )
        loss_cls = criterion(logits, batch_labels)
        # 只保留分类损失；rank-1 投影是确定性激活干预，非 loss 项
        loss = loss_cls

        optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数（对所有 requires_grad 且 grad 非空的参数）
        grad_norm = torch.sqrt(
            sum(p.grad.detach().pow(2).sum()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None)
        ).item()

        optimizer.step()

        # 定期记录几何特征
        if (step + 1) % record_every == 0:
            model.eval()

            # 在固定 probe set 上计算几何特征
            probe_results = compute_layer_metrics(
                model, probe_input_ids, probe_attention_mask, last_values,
                alpha=alpha, proj_active=proj_active, unit_vec=unit_vec,
            )

            with torch.no_grad():
                val_logits, _ = model(
                    val_input_ids, val_attention_mask,
                    alpha=alpha, proj_active=proj_active, unit_vec=unit_vec,
                )
                val_loss = criterion(val_logits, val_labels)

            ts = "step_" + str(step + 1)
            for (layer_name, eff, rep, col) in probe_results:
                write_metrics_row(
                    writer, seed, condition, ts, layer_name,
                    float(loss_cls.item()), float(val_loss.item()),
                    float(grad_norm), 2e-5, eff, rep, col,
                )
            file_obj.flush()

            # 打印训练进度
            print(
                f"Step {step + 1}/{steps} | "
                f"loss={loss_cls.item():.4f} | "
                f"val_loss={val_loss.item():.4f}"
            )

    file_obj.close()


def run_smoke_test():
    import collections
    # 生成数据集
    texts_train, labels_train, texts_val, labels_val, texts_probe, labels_probe = make_structured_dataset(data_seed=0)
    # 打印各 split 规模
    print(f"[Smoke Test] Train size: {len(texts_train)}")
    print(f"[Smoke Test] Val size:   {len(texts_val)}")
    print(f"[Smoke Test] Probe size: {len(texts_probe)}")
    # 各类别分布
    train_dist = collections.Counter(labels_train)
    val_dist = collections.Counter(labels_val)
    probe_dist = collections.Counter(labels_probe)
    print(f"[Smoke Test] Train class distribution: {dict(train_dist)}")
    print(f"[Smoke Test] Val class distribution:   {dict(val_dist)}")
    print(f"[Smoke Test] Probe class distribution: {dict(probe_dist)}")
    # 每类第一条 train 样本
    seen = set()
    for text, label in zip(texts_train, labels_train):
        if label not in seen:
            print(f"[Smoke Test] Train class {label} first sample: {text}")
            seen.add(label)
    # val 与 probe 首条
    print(f"[Smoke Test] Val first sample:   text={texts_val[0]!r}, label={labels_val[0]}")
    print(f"[Smoke Test] Probe first sample: text={texts_probe[0]!r}, label={labels_probe[0]}")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", type=str, choices=["normal", "condition_b"], default="normal")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--record-every", type=int, default=3)
    parser.add_argument("--probe-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        run_smoke_test()
        sys.exit(0)

    # 设备解析
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

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

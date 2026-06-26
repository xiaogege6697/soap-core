"""AI 训练日志示例数据生成器（合成数据，SOAP-Core v0.6 应用适配演示；支持 normal / diverging / overfit / mode_collapse）。"""

import csv
import math
import random
from pathlib import Path


def generate_training_run(path, length=500, mode="normal", seed=42):
    """生成训练日志 CSV 文件。"""
    rng = random.Random(seed)
    path = Path(path)
    
    # 转折点索引
    if mode == "normal":
        turning_point = length  # 无实际转折，但用于统一处理
    elif mode == "diverging" or mode == "overfit":
        turning_point = int(0.6 * length)
    elif mode == "mode_collapse":
        turning_point = int(0.4 * length)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # 用于存储前一步值（对于需要递归计算的模式）
    current_loss = None
    current_grad_norm = None
    current_val_loss = None
    
    rows = []
    
    for i in range(length):
        timestamp = f"step_{i}"
        
        if mode == "normal":
            loss = max(0.05, 2.3 * math.exp(-3.0 * i / length) + rng.gauss(0, 0.02))
            grad_norm = abs(0.3 + rng.gauss(0, 0.05))
            learning_rate = 1e-3
            val_loss = loss * 1.15 + rng.gauss(0, 0.03)
        
        elif mode == "diverging":
            if i < turning_point:
                # 前 60% 步：同 normal
                loss = max(0.05, 2.3 * math.exp(-3.0 * i / length) + rng.gauss(0, 0.02))
                grad_norm = abs(0.3 + rng.gauss(0, 0.05))
                learning_rate = 1e-3
                val_loss = loss * 1.15 + rng.gauss(0, 0.03)
                current_loss = loss
                current_grad_norm = grad_norm
            else:
                # 后 40% 步：递归计算
                if i == turning_point:
                    # 初始化转折点值
                    current_loss = current_loss  # 使用前一步的 loss
                    current_grad_norm = 0.3  # 从 0.3 起
                else:
                    # 基于前一步值
                    pass
                
                # 渐增噪声：sigma 从 0.02 开始，每步增加 0.001
                step_in_diverging = i - turning_point
                sigma_loss = 0.02 + 0.001 * step_in_diverging
                sigma_val = 0.02 + 0.001 * step_in_diverging
                
                loss = current_loss * 1.04 + rng.gauss(0, sigma_loss)
                grad_norm = current_grad_norm * 1.05  # 无噪声或固定噪声，按描述未指定噪声
                learning_rate = 1e-3
                val_loss = loss * 1.2 + rng.gauss(0, sigma_val)
                
                current_loss = loss
                current_grad_norm = grad_norm
        
        elif mode == "overfit":
            if i < turning_point:
                # 前 60% 步：同 normal
                loss = max(0.05, 2.3 * math.exp(-3.0 * i / length) + rng.gauss(0, 0.02))
                grad_norm = abs(0.3 + rng.gauss(0, 0.05))
                learning_rate = 1e-3
                val_loss = loss * 1.15 + rng.gauss(0, 0.03)
                current_val_loss = val_loss  # 存储转折点 val_loss
            else:
                # 后 40% 步：train 继续按 normal 公式衰减
                loss = max(0.05, 2.3 * math.exp(-3.0 * i / length) + rng.gauss(0, 0.02))
                grad_norm = abs(0.3 + rng.gauss(0, 0.05))
                learning_rate = 1e-3
                
                if i == turning_point:
                    # 从转折点 train*1.15 起
                    current_val_loss = loss * 1.15
                else:
                    # 基于前一步 val_loss
                    current_val_loss = current_val_loss * 1.015 + rng.gauss(0, 0.03)
                
                val_loss = current_val_loss
        
        elif mode == "mode_collapse":
            if i < turning_point:
                # 前 40% 步：同 normal 衰减
                loss = max(0.05, 2.3 * math.exp(-3.0 * i / length) + rng.gauss(0, 0.02))
                grad_norm = abs(0.3 + rng.gauss(0, 0.05))
                learning_rate = 1e-3
                val_loss = loss * 1.15 + rng.gauss(0, 0.02)
                current_loss = loss
                current_grad_norm = grad_norm
            else:
                # 后 60% 步：坍缩相
                step_in_collapse = i - turning_point
                total_collapse_steps = int(0.6 * length)
                
                if i == turning_point:
                    # 使用转折点值
                    pass
                else:
                    # 基于前一步值
                    pass
                
                # loss 平台化：每步乘 0.999，加小噪声
                loss = current_loss * 0.999 + rng.gauss(0, 0.005)
                
                # grad_norm 逐渐变小且异常平滑：每步乘 0.98，噪声 sigma 线性衰减
                sigma_grad = 0.05 * (1 - step_in_collapse / total_collapse_steps)  # 从 0.05 衰减到接近 0
                grad_norm = abs(current_grad_norm * 0.98 + rng.gauss(0, sigma_grad))
                
                learning_rate = 1e-3
                # val_loss 改善有限且轻微震荡：= train*1.15 + gauss(0,0.02)
                val_loss = loss * 1.15 + rng.gauss(0, 0.02)
                
                current_loss = loss
                current_grad_norm = grad_norm
        
        rows.append([timestamp, loss, grad_norm, learning_rate, val_loss])
    
    # 写入 CSV
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'loss', 'grad_norm', 'learning_rate', 'val_loss'])
        writer.writerows(rows)
    
    return path


if __name__ == "__main__":
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    modes = ["normal", "diverging", "overfit", "mode_collapse"]
    for mode in modes:
        file_path = examples_dir / f"synthetic_training_{mode}.csv"
        generate_training_run(file_path, mode=mode)
        print(f"Generated: {file_path}")

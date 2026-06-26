"""训练日志 CSV 适配进 SOAP-Core 数据层（v0.6 应用适配层；不改核心 loader，仅复用）。"""


def load_training_run(path):
    """复用核心 loader 加载训练日志 CSV。
    
    Returns:
        与 soap.data.loader.load_csv 返回类型完全一致，自动兼容现有 CLI / preprocessing.standardize。
    """
    from soap.data.loader import load_csv
    return load_csv(path)


def describe_training_run(path):
    """读取训练日志 CSV 并返回描述信息。
    
    Args:
        path: CSV 文件路径。
        
    Returns:
        dict: 包含 'columns' (首行列名列表) 和 'rows' (数据行数) 的字典。
        
    Raises:
        FileNotFoundError: 当 path 指定的文件不存在时。
    """
    import csv
    import os
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            columns = next(reader)
        except StopIteration:
            return {"columns": [], "rows": 0}
        
        rows = sum(1 for _ in reader)
    
    return {"columns": columns, "rows": rows}

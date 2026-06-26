from __future__ import annotations


def standardize(values: list[list[float]]) -> list[list[float]]:
    columns = list(zip(*values))
    means = [sum(column) / len(column) for column in columns]
    stds = []
    for column, mean in zip(columns, means):
        variance = sum((item - mean) ** 2 for item in column) / max(1, len(column) - 1)
        stds.append(variance ** 0.5 or 1.0)

    scaled = []
    for row in values:
        scaled.append([(item - mean) / std for item, mean, std in zip(row, means, stds)])
    return scaled


def sliding_windows(values: list[list[float]], window_size: int) -> list[list[float]]:
    if window_size < 1:
        raise ValueError("window_size 必须 >= 1。")
    if len(values) <= window_size:
        raise ValueError("数据长度必须大于 window_size。")

    windows = []
    for start in range(len(values) - window_size + 1):
        flattened = []
        for row in values[start : start + window_size]:
            flattened.extend(row)
        windows.append(flattened)
    return windows


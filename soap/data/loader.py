from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimeSeries:
    headers: list[str]
    timestamps: list[str]
    values: list[list[float]]


def load_csv(path: str | Path) -> TimeSeries:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        rows = [row for row in reader if row]

    if len(rows) < 3:
        raise ValueError("CSV 至少需要表头和两行数据。")

    raw_headers = [cell.strip() for cell in rows[0]]
    data_rows = rows[1:]

    first_column_numeric = _is_float(data_rows[0][0])
    if first_column_numeric:
        headers = raw_headers
        timestamps = [str(index) for index in range(len(data_rows))]
        values = [[float(cell) for cell in row] for row in data_rows]
    else:
        headers = raw_headers[1:]
        timestamps = [row[0] for row in data_rows]
        values = [[float(cell) for cell in row[1:]] for row in data_rows]

    if not headers:
        raise ValueError("CSV 需要至少一个数值特征列。")

    return TimeSeries(headers=headers, timestamps=timestamps, values=values)


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


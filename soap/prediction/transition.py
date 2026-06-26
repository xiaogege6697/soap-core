from __future__ import annotations


def transition_matrix(labels: list[int], cluster_count: int) -> list[list[float]]:
    counts = [[0 for _ in range(cluster_count)] for _ in range(cluster_count)]
    for current_label, next_label in zip(labels, labels[1:]):
        counts[current_label][next_label] += 1

    matrix = []
    for row in counts:
        total = sum(row)
        if total == 0:
            matrix.append([1.0 / cluster_count for _ in range(cluster_count)])
        else:
            matrix.append([value / total for value in row])
    return matrix


def next_state_probabilities(labels: list[int], matrix: list[list[float]]) -> list[float]:
    if not labels:
        raise ValueError("labels 不能为空。")
    return matrix[labels[-1]]


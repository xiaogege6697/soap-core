from __future__ import annotations


def kmeans(points: list[list[float]], k: int, iterations: int = 50) -> tuple[list[int], list[list[float]]]:
    if k < 1:
        raise ValueError("k 必须 >= 1。")
    if len(points) < k:
        raise ValueError("点数量必须 >= k。")

    centroids = [points[index][:] for index in _initial_indices(len(points), k)]
    labels = [0 for _ in points]

    for _ in range(iterations):
        labels = [_nearest(point, centroids) for point in points]
        new_centroids = []
        for cluster_id in range(k):
            members = [point for point, label in zip(points, labels) if label == cluster_id]
            if not members:
                new_centroids.append(centroids[cluster_id])
            else:
                new_centroids.append(_mean_vector(members))
        if new_centroids == centroids:
            break
        centroids = new_centroids

    return labels, centroids


def _initial_indices(length: int, k: int) -> list[int]:
    if k == 1:
        return [0]
    return [round(index * (length - 1) / (k - 1)) for index in range(k)]


def _nearest(point: list[float], centroids: list[list[float]]) -> int:
    distances = [_squared_distance(point, centroid) for centroid in centroids]
    return min(range(len(distances)), key=lambda index: distances[index])


def _mean_vector(points: list[list[float]]) -> list[float]:
    return [sum(column) / len(column) for column in zip(*points)]


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


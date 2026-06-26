from math import isfinite, sqrt
from typing import Sequence


NumberSeries = Sequence[float] | Sequence[Sequence[float]]


def estimate_embedding_dimension_fnn(
    values: list[float] | list[list[float]],
    delay: int,
    max_dim: int,
    tolerance: float = 10.0,
    threshold: float = 0.05,
) -> int:
    """Estimate embedding dimension with the false nearest neighbors ratio."""
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if max_dim < 1:
        raise ValueError("max_dim must be >= 1")
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    observations = _normalize_observations(values)

    for dimension in range(1, max_dim + 1):
        ratio = _false_nearest_neighbor_ratio(observations, delay, dimension, tolerance)
        if ratio < threshold:
            return dimension

    return max_dim


def _normalize_observations(values: NumberSeries) -> list[tuple[float, ...]]:
    if not values:
        raise ValueError("values must not be empty")

    observations: list[tuple[float, ...]] = []
    first_value = values[0]

    if isinstance(first_value, (int, float)):
        for value in values:
            if not isinstance(value, (int, float)):
                raise ValueError("values must be consistently scalar or vector observations")
            numeric_value = float(value)
            if not isfinite(numeric_value):
                raise ValueError("values must contain only finite numbers")
            observations.append((numeric_value,))
        return observations

    feature_count = len(first_value)
    if feature_count == 0:
        raise ValueError("vector observations must not be empty")

    for row in values:
        if isinstance(row, (int, float)) or len(row) != feature_count:
            raise ValueError("values must be consistently sized vector observations")
        numeric_row = tuple(float(value) for value in row)
        if any(not isfinite(value) for value in numeric_row):
            raise ValueError("values must contain only finite numbers")
        observations.append(numeric_row)

    return observations


def _false_nearest_neighbor_ratio(
    observations: list[tuple[float, ...]],
    delay: int,
    dimension: int,
    tolerance: float,
) -> float:
    embedded = _takens_vectors(observations, delay, dimension + 1)
    if len(embedded) < 2:
        return 0.0

    current_width = dimension * len(observations[0])
    false_neighbors = 0

    for index, vector in enumerate(embedded):
        current_vector = vector[:current_width]
        neighbor_index, neighbor_distance = _nearest_neighbor(
            current_vector,
            embedded,
            current_width,
            index,
        )
        next_distance = _euclidean_distance(
            vector[current_width:],
            embedded[neighbor_index][current_width:],
        )
        if neighbor_distance == 0:
            is_false_neighbor = next_distance > 0
        else:
            is_false_neighbor = next_distance / neighbor_distance > tolerance
        if is_false_neighbor:
            false_neighbors += 1

    return false_neighbors / len(embedded)


def _takens_vectors(
    observations: list[tuple[float, ...]],
    delay: int,
    dimension: int,
) -> list[tuple[float, ...]]:
    vector_count = len(observations) - (dimension - 1) * delay
    if vector_count <= 0:
        return []

    vectors: list[tuple[float, ...]] = []
    for start in range(vector_count):
        coordinates: list[float] = []
        for offset in range(dimension):
            coordinates.extend(observations[start + offset * delay])
        vectors.append(tuple(coordinates))
    return vectors


def _nearest_neighbor(
    current_vector: tuple[float, ...],
    embedded: list[tuple[float, ...]],
    current_width: int,
    current_index: int,
) -> tuple[int, float]:
    nearest_index = -1
    nearest_distance = float("inf")

    for index, vector in enumerate(embedded):
        if index == current_index:
            continue
        distance = _euclidean_distance(current_vector, vector[:current_width])
        if distance < nearest_distance:
            nearest_index = index
            nearest_distance = distance

    return nearest_index, nearest_distance


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))

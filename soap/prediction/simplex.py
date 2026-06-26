from __future__ import annotations

import math


def simplex_predict(
    embedding: list[list[float]],
    target: list[list[float]] | None = None,
    neighbors: int | None = None,
) -> list[list[float]]:
    """Predict one step ahead with simplex nearest-neighbor averaging."""
    states = _validate_matrix(embedding, "embedding")
    target_states = states if target is None else _validate_matrix(target, "target")

    if len(target_states) != len(states):
        raise ValueError("target length must match embedding length")
    if len(states) < 3:
        raise ValueError("simplex prediction needs at least 3 states")

    dimension = len(states[0])
    neighbor_count = dimension + 1 if neighbors is None else neighbors
    if neighbor_count < 1:
        raise ValueError("neighbors must be >= 1")

    library_count = len(states) - 1
    available_neighbors = library_count - 1
    if neighbor_count > available_neighbors:
        raise ValueError(
            "not enough states for simplex prediction: "
            f"need at least {neighbor_count + 2} states for {neighbor_count} neighbors, "
            f"got {len(states)}"
        )

    predictions: list[list[float]] = []
    for query_index, query_state in enumerate(states[:-1]):
        neighbor_indices = _nearest_neighbor_indices(
            states=states,
            query_state=query_state,
            query_index=query_index,
            library_count=library_count,
            neighbor_count=neighbor_count,
        )
        predictions.append(
            _weighted_average_next_state(
                states=states,
                target=target_states,
                neighbor_indices=neighbor_indices,
                query_state=query_state,
            )
        )

    return predictions


def prediction_skill(
    predicted: list[list[float]], actual: list[list[float]]
) -> dict[str, float]:
    """Return RMSE, MAE, and variance-normalized prediction skill."""
    predicted_states = _validate_matrix(predicted, "predicted")
    actual_states = _validate_matrix(actual, "actual")

    if len(predicted_states) != len(actual_states):
        raise ValueError("predicted length must match actual length")
    if len(predicted_states[0]) != len(actual_states[0]):
        raise ValueError("predicted dimension must match actual dimension")

    squared_error = 0.0
    absolute_error = 0.0
    value_count = 0
    actual_values: list[float] = []

    for predicted_row, actual_row in zip(predicted_states, actual_states):
        if len(predicted_row) != len(actual_row):
            raise ValueError("predicted dimension must match actual dimension")
        for predicted_value, actual_value in zip(predicted_row, actual_row):
            error = predicted_value - actual_value
            squared_error += error * error
            absolute_error += abs(error)
            value_count += 1
            actual_values.append(actual_value)

    mse = squared_error / value_count
    mae = absolute_error / value_count
    rmse = math.sqrt(mse)
    baseline_mse = _mean_baseline_mse(actual_values)
    skill = 1.0 - mse / baseline_mse if baseline_mse > 0.0 else (1.0 if mse == 0.0 else 0.0)

    return {"rmse": rmse, "mae": mae, "skill": skill}


def _validate_matrix(values: list[list[float]], name: str) -> list[list[float]]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if not all(isinstance(row, list) and row for row in values):
        raise ValueError(f"{name} must contain non-empty rows")

    width = len(values[0])
    matrix: list[list[float]] = []
    for row in values:
        if len(row) != width:
            raise ValueError(f"{name} rows must have consistent dimensions")
        try:
            matrix.append([float(value) for value in row])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numeric values") from error
    return matrix


def _nearest_neighbor_indices(
    states: list[list[float]],
    query_state: list[float],
    query_index: int,
    library_count: int,
    neighbor_count: int,
) -> list[int]:
    distances = [
        (_euclidean_distance(query_state, candidate_state), candidate_index)
        for candidate_index, candidate_state in enumerate(states[:library_count])
        if candidate_index != query_index
    ]
    distances.sort(key=lambda item: (item[0], item[1]))
    return [candidate_index for _, candidate_index in distances[:neighbor_count]]


def _weighted_average_next_state(
    states: list[list[float]],
    target: list[list[float]],
    neighbor_indices: list[int],
    query_state: list[float],
) -> list[float]:
    distances = [
        _euclidean_distance(query_state, states[neighbor_index])
        for neighbor_index in neighbor_indices
    ]
    zero_distance_indices = [
        neighbor_index
        for neighbor_index, distance in zip(neighbor_indices, distances)
        if distance == 0.0
    ]
    if zero_distance_indices:
        return _average_rows([target[neighbor_index + 1] for neighbor_index in zero_distance_indices])

    nearest_distance = distances[0]
    weights = [math.exp(-distance / nearest_distance) for distance in distances]
    total_weight = sum(weights)
    output_dimension = len(target[0])

    prediction = [0.0 for _ in range(output_dimension)]
    for neighbor_index, weight in zip(neighbor_indices, weights):
        next_state = target[neighbor_index + 1]
        for value_index, value in enumerate(next_state):
            prediction[value_index] += weight * value

    return [value / total_weight for value in prediction]


def _average_rows(rows: list[list[float]]) -> list[float]:
    output_dimension = len(rows[0])
    totals = [0.0 for _ in range(output_dimension)]
    for row in rows:
        for value_index, value in enumerate(row):
            totals[value_index] += value
    return [value / len(rows) for value in totals]


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def _mean_baseline_mse(values: list[float]) -> float:
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)

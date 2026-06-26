from __future__ import annotations

import math

import numpy as np


def smap_predict(
    embedding: list[list[float]],
    theta: float = 2.0,
    neighbors: int | None = None,
) -> list[list[float]]:
    states = _validate_embedding(embedding)
    if theta < 0:
        raise ValueError("theta must be >= 0")

    row_count, dimension = states.shape
    if row_count < 3:
        raise ValueError("embedding need at least 3 rows for S-Map prediction")

    candidate_count = row_count - 1
    available_neighbors = candidate_count - 1
    if neighbors is None:
        neighbor_count = available_neighbors
    else:
        if neighbors < 1:
            raise ValueError("neighbors must be >= 1")
        if neighbors > available_neighbors:
            raise ValueError(
                f"neighbors must be <= {available_neighbors} for this embedding"
            )
        neighbor_count = neighbors

    predictions = []
    candidate_indexes = np.arange(candidate_count)
    candidate_states = states[:candidate_count]
    candidate_next_states = states[1:]

    for state_index in range(candidate_count):
        usable_indexes = candidate_indexes[candidate_indexes != state_index]
        usable_states = candidate_states[usable_indexes]
        distances = np.linalg.norm(usable_states - states[state_index], axis=1)
        nearest_positions = np.argsort(distances)[:neighbor_count]
        local_indexes = usable_indexes[nearest_positions]
        local_distances = distances[nearest_positions]
        weights = _distance_weights(local_distances, theta)

        local_states = candidate_states[local_indexes]
        local_next_states = candidate_next_states[local_indexes]
        predicted = _local_linear_prediction(
            local_states=local_states,
            local_next_states=local_next_states,
            query_state=states[state_index],
            weights=weights,
            dimension=dimension,
        )
        predictions.append(predicted.tolist())

    return predictions


def smap_skill(predicted: list[list[float]], actual: list[list[float]]) -> dict[str, float]:
    predicted_states = _validate_embedding(predicted, name="predicted")
    actual_states = _validate_embedding(actual, name="actual")
    if predicted_states.shape != actual_states.shape:
        raise ValueError("predicted and actual must have the same shape")

    errors = predicted_states - actual_states
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(float(np.mean(errors**2))))

    predicted_flat = predicted_states.reshape(-1)
    actual_flat = actual_states.reshape(-1)
    if predicted_flat.size < 2:
        rho = 0.0
    else:
        predicted_std = float(np.std(predicted_flat))
        actual_std = float(np.std(actual_flat))
        if predicted_std == 0.0 or actual_std == 0.0:
            rho = 0.0
        else:
            rho = float(np.corrcoef(predicted_flat, actual_flat)[0, 1])

    return {"mae": mae, "rmse": rmse, "rho": rho}


def _validate_embedding(
    embedding: list[list[float]],
    name: str = "embedding",
) -> np.ndarray:
    if not embedding:
        raise ValueError(f"{name} must not be empty")
    if any(not row for row in embedding):
        raise ValueError(f"{name} rows must not be empty")

    dimension = len(embedding[0])
    if any(len(row) != dimension for row in embedding):
        raise ValueError(f"{name} rows must have the same dimension")

    try:
        states = np.asarray(embedding, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error

    if states.ndim != 2:
        raise ValueError(f"{name} must be a 2D list")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"{name} must contain finite values")
    return states


def _distance_weights(distances: np.ndarray, theta: float) -> np.ndarray:
    if theta == 0:
        return np.ones_like(distances, dtype=float)

    mean_distance = float(np.mean(distances))
    if mean_distance == 0.0:
        return np.ones_like(distances, dtype=float)
    return np.exp(-theta * distances / mean_distance)


def _local_linear_prediction(
    local_states: np.ndarray,
    local_next_states: np.ndarray,
    query_state: np.ndarray,
    weights: np.ndarray,
    dimension: int,
) -> np.ndarray:
    if local_states.shape[0] < dimension + 1:
        return _weighted_average(local_next_states, weights)

    design = np.column_stack([np.ones(local_states.shape[0]), local_states])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_targets = local_next_states * np.sqrt(weights)[:, None]

    try:
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_design,
            weighted_targets,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return _weighted_average(local_next_states, weights)

    if rank < dimension + 1:
        return _weighted_average(local_next_states, weights)

    query_design = np.concatenate([[1.0], query_state])
    return query_design @ coefficients


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total_weight = float(np.sum(weights))
    if total_weight == 0.0:
        return np.mean(values, axis=0)
    return np.average(values, axis=0, weights=weights)

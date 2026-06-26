from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionScore:
    dimension: int
    reconstruction_error: float
    prediction_error: float
    complexity_penalty: float
    total_score: float


def score_dimensions(
    states: list[list[float]],
    max_dim: int,
    complexity_weight: float = 0.03,
) -> list[DimensionScore]:
    if not states:
        raise ValueError("states 不能为空。")

    feature_count = len(states[0])
    upper_dim = min(max_dim, feature_count)
    scores: list[DimensionScore] = []

    variances = _column_variances(states)
    ranked_indices = sorted(range(feature_count), key=lambda index: variances[index], reverse=True)
    total_variance = sum(variances) or 1.0

    for dimension in range(1, upper_dim + 1):
        selected = ranked_indices[:dimension]
        reconstruction_error = 1.0 - (sum(variances[index] for index in selected) / total_variance)
        embedded = project_by_indices(states, selected)
        prediction_error = one_step_prediction_error(embedded)
        complexity_penalty = complexity_weight * dimension
        total_score = prediction_error + reconstruction_error + complexity_penalty
        scores.append(
            DimensionScore(
                dimension=dimension,
                reconstruction_error=reconstruction_error,
                prediction_error=prediction_error,
                complexity_penalty=complexity_penalty,
                total_score=total_score,
            )
        )
    return scores


def select_optimal_dimension(scores: list[DimensionScore]) -> DimensionScore:
    if not scores:
        raise ValueError("scores 不能为空。")
    return min(scores, key=lambda score: score.total_score)


def project_to_dimension(states: list[list[float]], dimension: int) -> list[list[float]]:
    variances = _column_variances(states)
    ranked_indices = sorted(range(len(states[0])), key=lambda index: variances[index], reverse=True)
    return project_by_indices(states, ranked_indices[:dimension])


def project_by_indices(states: list[list[float]], indices: list[int]) -> list[list[float]]:
    return [[row[index] for index in indices] for row in states]


def one_step_prediction_error(embedded: list[list[float]]) -> float:
    if len(embedded) < 3:
        return 1.0

    errors = []
    for index in range(1, len(embedded) - 1):
        predicted = embedded[index]
        actual = embedded[index + 1]
        errors.append(_squared_distance(predicted, actual))
    baseline = _mean_pairwise_step(embedded) or 1.0
    return (sum(errors) / len(errors)) / baseline


def _column_variances(states: list[list[float]]) -> list[float]:
    columns = list(zip(*states))
    variances = []
    for column in columns:
        mean = sum(column) / len(column)
        variances.append(sum((item - mean) ** 2 for item in column) / max(1, len(column) - 1))
    return variances


def _mean_pairwise_step(embedded: list[list[float]]) -> float:
    distances = [
        _squared_distance(embedded[index], embedded[index + 1])
        for index in range(len(embedded) - 1)
    ]
    return sum(distances) / len(distances)


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


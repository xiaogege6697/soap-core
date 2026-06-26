"""Recurrence diagnostics for embedded point clouds."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, isfinite, sqrt


def recurrence_matrix(
    points: list[list[float]],
    radius: float | None = None,
    quantile: float = 0.1,
) -> list[list[int]]:
    """Build a binary recurrence matrix from point distances."""
    _validate_radius_and_quantile(radius, quantile)
    observations = _normalize_points(points)
    if not observations:
        return []

    selected_radius = radius
    if selected_radius is None:
        selected_radius = _distance_quantile(observations, quantile)

    matrix: list[list[int]] = []
    for left in observations:
        row: list[int] = []
        for right in observations:
            row.append(1 if _euclidean_distance(left, right) <= selected_radius else 0)
        matrix.append(row)

    return matrix


def recurrence_summary(matrix: list[list[int]]) -> dict[str, float]:
    """Summarize recurrence density and diagonal-line structure."""
    _validate_matrix(matrix)
    size = len(matrix)
    if size == 0:
        return {
            "recurrence_rate": 0.0,
            "determinism_proxy": 0.0,
            "average_diagonal_length_proxy": 0.0,
        }

    recurrent_points = sum(sum(row) for row in matrix)
    diagonal_lengths = _off_diagonal_line_lengths(matrix, minimum_length=2)
    deterministic_points = sum(diagonal_lengths)

    off_diagonal_recurrent_points = recurrent_points - size
    determinism_proxy = (
        deterministic_points / off_diagonal_recurrent_points
        if off_diagonal_recurrent_points > 0
        else 0.0
    )
    average_diagonal_length_proxy = (
        deterministic_points / len(diagonal_lengths) if diagonal_lengths else 0.0
    )

    return {
        "recurrence_rate": recurrent_points / (size * size),
        "determinism_proxy": determinism_proxy,
        "average_diagonal_length_proxy": average_diagonal_length_proxy,
    }


def _validate_radius_and_quantile(radius: float | None, quantile: float) -> None:
    if radius is not None:
        numeric_radius = float(radius)
        if not isfinite(numeric_radius) or numeric_radius < 0:
            raise ValueError("radius must be a finite non-negative number")
    numeric_quantile = float(quantile)
    if not isfinite(numeric_quantile) or not 0 < numeric_quantile <= 1:
        raise ValueError("quantile must be in the interval (0, 1]")


def _normalize_points(points: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    observations: list[tuple[float, ...]] = []
    feature_count: int | None = None

    for point in points:
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence):
            raise ValueError("points must be a sequence of vector observations")
        if len(point) == 0:
            raise ValueError("point vectors must not be empty")
        if feature_count is None:
            feature_count = len(point)
        elif len(point) != feature_count:
            raise ValueError("point vectors must have consistent dimensions")

        numeric_point = tuple(float(value) for value in point)
        if any(not isfinite(value) for value in numeric_point):
            raise ValueError("points must contain only finite numbers")
        observations.append(numeric_point)

    return observations


def _distance_quantile(points: list[tuple[float, ...]], quantile: float) -> float:
    distances: list[float] = []
    for left_index, left in enumerate(points):
        for right in points[left_index + 1 :]:
            distances.append(_euclidean_distance(left, right))

    if not distances:
        return 0.0

    distances.sort()
    selected_index = min(len(distances) - 1, max(0, ceil(len(distances) * quantile) - 1))
    return distances[selected_index]


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def _validate_matrix(matrix: Sequence[Sequence[int]]) -> None:
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            raise ValueError("matrix must be square")
        for value in row:
            if value not in (0, 1):
                raise ValueError("matrix must contain only 0/1 values")


def _off_diagonal_line_lengths(matrix: Sequence[Sequence[int]], minimum_length: int) -> list[int]:
    size = len(matrix)
    lengths: list[int] = []

    for offset in range(-(size - 1), size):
        if offset == 0:
            continue
        run_length = 0
        for row_index in range(size):
            column_index = row_index + offset
            if not 0 <= column_index < size:
                continue
            if matrix[row_index][column_index] == 1:
                run_length += 1
                continue
            if run_length >= minimum_length:
                lengths.append(run_length)
            run_length = 0
        if run_length >= minimum_length:
            lengths.append(run_length)

    return lengths

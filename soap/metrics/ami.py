"""Average mutual information based delay estimation."""

from __future__ import annotations

import math
from collections.abc import Sequence


NumberSeries = Sequence[float] | Sequence[Sequence[float]]


def estimate_delay_ami(values: NumberSeries, max_delay: int, bins: int = 16) -> int:
    """Estimate delay from the first local minimum of mutual information.

    ``values`` may be a univariate sequence or a multivariate sequence of rows.
    For multivariate rows, only the first observation column is used so the
    delay estimate is tied to the primary measured signal.

    Args:
        values: Time ordered scalar values or rows of scalar observations.
        max_delay: Largest positive delay to evaluate.
        bins: Number of equal-width bins used to discretize values.

    Returns:
        The selected delay in ``[1, max_delay]``.

    Raises:
        ValueError: If parameters are invalid or the series is too short.
    """

    if max_delay < 1:
        raise ValueError("max_delay must be at least 1")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    series = _as_primary_series(values)
    if len(series) <= max_delay:
        raise ValueError("values must contain more samples than max_delay")

    discretized = _discretize(series, bins)
    scores = [
        _mutual_information(discretized[:-delay], discretized[delay:], bins)
        for delay in range(1, max_delay + 1)
    ]

    selected_index = _first_local_minimum_index(scores)
    if selected_index is None:
        selected_index = min(range(len(scores)), key=scores.__getitem__)

    return selected_index + 1


def _as_primary_series(values: NumberSeries) -> list[float]:
    series: list[float] = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                raise ValueError("multivariate rows must not be empty")
            series.append(float(value[0]))
        else:
            series.append(float(value))
    if len(series) < 2:
        raise ValueError("values must contain at least two samples")
    return series


def _discretize(series: Sequence[float], bins: int) -> list[int]:
    minimum = min(series)
    maximum = max(series)
    if minimum == maximum:
        return [0 for _ in series]

    width = (maximum - minimum) / bins
    return [min(bins - 1, int((value - minimum) / width)) for value in series]


def _mutual_information(
    left_values: Sequence[int], right_values: Sequence[int], bins: int
) -> float:
    sample_count = len(left_values)
    left_counts = [0] * bins
    right_counts = [0] * bins
    joint_counts = [[0] * bins for _ in range(bins)]

    for left_value, right_value in zip(left_values, right_values):
        left_counts[left_value] += 1
        right_counts[right_value] += 1
        joint_counts[left_value][right_value] += 1

    information = 0.0
    for left_bin in range(bins):
        for right_bin in range(bins):
            joint_count = joint_counts[left_bin][right_bin]
            if joint_count == 0:
                continue
            information += (joint_count / sample_count) * math.log(
                (joint_count * sample_count)
                / (left_counts[left_bin] * right_counts[right_bin])
            )
    return information


def _first_local_minimum_index(scores: Sequence[float]) -> int | None:
    for index in range(1, len(scores) - 1):
        if scores[index] < scores[index - 1] and scores[index] <= scores[index + 1]:
            return index
    return None

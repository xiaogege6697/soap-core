import math
import random

import pytest

from soap.metrics.fnn import estimate_embedding_dimension_fnn


def test_fnn_stops_early_for_simple_periodic_sequence():
    values = [math.sin(index * 0.2) for index in range(120)]

    dimension = estimate_embedding_dimension_fnn(
        values,
        delay=2,
        max_dim=6,
        tolerance=10.0,
        threshold=0.2,
    )

    assert 1 <= dimension < 6


def test_fnn_returns_dimension_in_range_for_random_sequence():
    random_generator = random.Random(7)
    values = [random_generator.random() for _ in range(80)]

    dimension = estimate_embedding_dimension_fnn(values, delay=1, max_dim=5)

    assert 1 <= dimension <= 5


def test_fnn_returns_dimension_in_range_for_monotonic_sequence():
    values = [float(index) for index in range(50)]

    dimension = estimate_embedding_dimension_fnn(values, delay=3, max_dim=4)

    assert 1 <= dimension <= 4


def test_fnn_returns_max_dim_when_ratio_never_below_threshold():
    values = [float(index % 5) for index in range(50)]

    dimension = estimate_embedding_dimension_fnn(values, delay=1, max_dim=4, threshold=0.0)

    assert dimension == 4


def test_fnn_supports_multivariate_observations():
    values = [[math.sin(index * 0.2), math.cos(index * 0.2)] for index in range(100)]

    dimension = estimate_embedding_dimension_fnn(values, delay=2, max_dim=5)

    assert 1 <= dimension <= 5


def test_fnn_rejects_invalid_delay():
    with pytest.raises(ValueError, match="delay"):
        estimate_embedding_dimension_fnn([1.0, 2.0, 3.0], delay=0, max_dim=3)


def test_fnn_rejects_invalid_max_dim():
    with pytest.raises(ValueError, match="max_dim"):
        estimate_embedding_dimension_fnn([1.0, 2.0, 3.0], delay=1, max_dim=0)

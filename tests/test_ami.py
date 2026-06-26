import math

import pytest

from soap.metrics import estimate_delay_ami


def test_estimate_delay_ami_returns_first_local_minimum_for_periodic_series():
    values = [math.sin(2 * math.pi * index / 20) for index in range(200)]

    delay = estimate_delay_ami(values, max_delay=12, bins=16)

    assert 3 <= delay <= 7


def test_estimate_delay_ami_uses_first_column_for_multivariate_series():
    values = [
        [math.sin(2 * math.pi * index / 20), float(index)]
        for index in range(200)
    ]

    delay = estimate_delay_ami(values, max_delay=12, bins=16)

    assert 3 <= delay <= 7


def test_estimate_delay_ami_rejects_short_series():
    with pytest.raises(ValueError, match="more samples than max_delay"):
        estimate_delay_ami([1.0, 2.0], max_delay=2)


@pytest.mark.parametrize(
    ("max_delay", "bins", "message"),
    [
        (0, 16, "max_delay"),
        (1, 1, "bins"),
    ],
)
def test_estimate_delay_ami_rejects_invalid_parameters(max_delay, bins, message):
    with pytest.raises(ValueError, match=message):
        estimate_delay_ami([1.0, 2.0, 3.0], max_delay=max_delay, bins=bins)

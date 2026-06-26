import pytest

from soap.metrics.recurrence import recurrence_matrix, recurrence_summary


def test_recurrence_matrix_is_symmetric_binary_for_simple_points():
    points = [[0.0], [0.5], [2.0]]

    matrix = recurrence_matrix(points, radius=0.75)

    assert matrix == [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]
    assert all(value in (0, 1) for row in matrix for value in row)
    assert matrix == [list(row) for row in zip(*matrix)]


def test_recurrence_summary_reports_reasonable_rates():
    matrix = [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
    ]

    summary = recurrence_summary(matrix)

    assert summary["recurrence_rate"] == pytest.approx(7 / 9)
    assert 0.0 <= summary["determinism_proxy"] <= 1.0
    assert summary["average_diagonal_length_proxy"] >= 0.0


def test_recurrence_matrix_selects_automatic_radius_from_quantile():
    points = [[0.0], [1.0], [3.0]]

    matrix = recurrence_matrix(points, quantile=0.5)

    assert matrix == [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
    ]


@pytest.mark.parametrize(
    ("radius", "quantile", "message"),
    [
        (-0.1, 0.1, "radius"),
        (float("inf"), 0.1, "radius"),
        (None, 0.0, "quantile"),
        (None, 1.1, "quantile"),
    ],
)
def test_recurrence_matrix_rejects_invalid_parameters(radius, quantile, message):
    with pytest.raises(ValueError, match=message):
        recurrence_matrix([[0.0], [1.0]], radius=radius, quantile=quantile)


def test_recurrence_matrix_rejects_inconsistent_points():
    with pytest.raises(ValueError, match="consistent dimensions"):
        recurrence_matrix([[0.0], [1.0, 2.0]], radius=1.0)


def test_recurrence_handles_empty_input():
    matrix = recurrence_matrix([])

    assert matrix == []
    assert recurrence_summary(matrix) == {
        "recurrence_rate": 0.0,
        "determinism_proxy": 0.0,
        "average_diagonal_length_proxy": 0.0,
    }

import pytest

from soap.prediction.smap import smap_predict, smap_skill


def test_smap_predicts_linear_sequence_next_states():
    embedding = [[float(index)] for index in range(8)]

    predicted = smap_predict(embedding, theta=2.0)
    actual = embedding[1:]
    skill = smap_skill(predicted, actual)

    assert len(predicted) == len(actual)
    assert skill["rmse"] == pytest.approx(0.0, abs=1e-10)
    assert skill["mae"] == pytest.approx(0.0, abs=1e-10)


def test_smap_predicts_periodic_sequence_reasonably():
    embedding = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ]

    predicted = smap_predict(embedding, theta=2.0, neighbors=3)
    actual = embedding[1:]
    skill = smap_skill(predicted, actual)

    assert len(predicted) == len(actual)
    assert skill["rmse"] < 0.8
    assert skill["rho"] > 0.5


def test_smap_theta_zero_uses_uniform_distance_weights():
    embedding = [[float(index)] for index in range(6)]

    predicted = smap_predict(embedding, theta=0.0, neighbors=2)

    assert len(predicted) == len(embedding) - 1
    assert all(len(row) == 1 for row in predicted)


def test_smap_rejects_negative_theta():
    with pytest.raises(ValueError, match="theta must be >= 0"):
        smap_predict([[0.0], [1.0], [2.0]], theta=-0.1)


def test_smap_reports_insufficient_data():
    with pytest.raises(ValueError, match="need at least 3 rows"):
        smap_predict([[0.0], [1.0]])


def test_smap_rejects_inconsistent_dimensions():
    with pytest.raises(ValueError, match="same dimension"):
        smap_predict([[0.0], [1.0, 2.0], [3.0]])


def test_smap_rejects_invalid_neighbors():
    with pytest.raises(ValueError, match="neighbors must be >= 1"):
        smap_predict([[0.0], [1.0], [2.0]], neighbors=0)


def test_smap_skill_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="same shape"):
        smap_skill([[1.0], [2.0]], [[1.0]])

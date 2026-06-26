import math

import pytest

from soap.prediction import prediction_skill, simplex_predict


def test_simplex_predicts_linear_sequence_one_step_ahead():
    embedding = [[float(index)] for index in range(10)]

    predicted = simplex_predict(embedding)
    actual = embedding[1:]
    skill = prediction_skill(predicted, actual)

    assert len(predicted) == len(actual)
    assert skill["rmse"] < 1.6
    assert skill["mae"] < 1.5
    assert "skill" in skill


def test_simplex_predicts_periodic_sequence_with_custom_neighbors():
    embedding = [
        [math.sin(2 * math.pi * index / 12), math.cos(2 * math.pi * index / 12)]
        for index in range(48)
    ]

    predicted = simplex_predict(embedding, neighbors=3)
    actual = embedding[1:]
    skill = prediction_skill(predicted, actual)

    assert len(predicted) == len(actual)
    assert skill["rmse"] < 0.35


def test_simplex_uses_default_dimension_plus_one_neighbors():
    embedding = [[float(index), float(index % 3)] for index in range(6)]

    predicted = simplex_predict(embedding)

    assert len(predicted) == len(embedding) - 1


def test_simplex_rejects_short_sequence_for_default_neighbors():
    with pytest.raises(ValueError, match="not enough states"):
        simplex_predict([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])


def test_simplex_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="embedding"):
        simplex_predict([])
    with pytest.raises(ValueError, match="target length"):
        simplex_predict([[0.0], [1.0], [2.0]], target=[[0.0], [1.0]], neighbors=1)
    with pytest.raises(ValueError, match="neighbors"):
        simplex_predict([[0.0], [1.0], [2.0]], neighbors=0)


def test_prediction_skill_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="predicted length"):
        prediction_skill([[1.0]], [[1.0], [2.0]])

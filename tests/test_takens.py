import pytest

from soap.embedding import takens_embedding


def test_takens_embedding_univariate_series():
    series = [[1.0], [2.0], [3.0], [4.0], [5.0]]

    embedded = takens_embedding(series, delay=1, dimension=3)

    assert embedded == [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ]


def test_takens_embedding_multivariate_series_flattens_delay_blocks():
    series = [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ]

    embedded = takens_embedding(series, delay=2, dimension=2)

    assert embedded == [
        [1.0, 10.0, 3.0, 30.0],
        [2.0, 20.0, 4.0, 40.0],
        [3.0, 30.0, 5.0, 50.0],
    ]


def test_takens_embedding_rejects_invalid_delay():
    with pytest.raises(ValueError, match="delay must be >= 1"):
        takens_embedding([[1.0], [2.0]], delay=0, dimension=1)


def test_takens_embedding_rejects_invalid_dimension():
    with pytest.raises(ValueError, match="dimension must be >= 1"):
        takens_embedding([[1.0], [2.0]], delay=1, dimension=0)


def test_takens_embedding_reports_insufficient_data():
    with pytest.raises(ValueError, match="need at least 5 rows"):
        takens_embedding([[1.0], [2.0], [3.0], [4.0]], delay=2, dimension=3)

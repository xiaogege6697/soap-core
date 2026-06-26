from soap.dimension.selector import score_dimensions, select_optimal_dimension


def test_score_dimensions_returns_valid_optimal_dimension():
    states = [[float(index), float(index % 3), float(index % 5)] for index in range(20)]
    scores = score_dimensions(states, max_dim=3)
    optimal = select_optimal_dimension(scores)
    assert 1 <= optimal.dimension <= 3


def test_pca_score_dimensions_returns_valid_optimal_dimension():
    from soap.dimension.pca_selector import score_dimensions_with_pca

    states = [
        [float(index), float(index % 3), float(index % 5), float(index % 7)]
        for index in range(30)
    ]
    scores = score_dimensions_with_pca(states, max_dim=4)
    optimal = select_optimal_dimension(scores)
    assert 1 <= optimal.dimension <= 4
    assert all(score.reconstruction_error >= 0 for score in scores)

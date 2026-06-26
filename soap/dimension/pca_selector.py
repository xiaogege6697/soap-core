from __future__ import annotations

from soap.dimension.selector import DimensionScore


def score_dimensions_with_pca(
    states: list[list[float]],
    max_dim: int,
    complexity_weight: float = 0.03,
    test_fraction: float = 0.25,
) -> list[DimensionScore]:
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
    except ImportError as error:
        raise RuntimeError("PCA method requires science dependencies: pip install -e '.[science]'") from error

    if not states:
        raise ValueError("states 不能为空。")
    if max_dim < 1:
        raise ValueError("max_dim 必须 >= 1。")
    if not 0.0 < test_fraction < 0.8:
        raise ValueError("test_fraction 必须在 0 和 0.8 之间。")

    data = np.asarray(states, dtype=float)
    sample_count, feature_count = data.shape
    if sample_count < 6:
        raise ValueError("PCA scoring 至少需要 6 个状态点。")

    upper_dim = min(max_dim, feature_count, sample_count - 2)
    split_index = int(sample_count * (1.0 - test_fraction))
    split_index = max(3, min(split_index, sample_count - 3))
    train = data[:split_index]
    test = data[split_index:]
    total_variance = float(np.var(test, axis=0).sum()) or 1.0

    scores: list[DimensionScore] = []
    for dimension in range(1, upper_dim + 1):
        pca = PCA(n_components=dimension, random_state=0)
        train_embedding = pca.fit_transform(train)
        test_embedding = pca.transform(test)

        reconstructed = pca.inverse_transform(test_embedding)
        reconstruction_error = float(np.mean((test - reconstructed) ** 2) / total_variance)

        model = LinearRegression()
        model.fit(train_embedding[:-1], train_embedding[1:])
        predicted = model.predict(test_embedding[:-1])
        actual = test_embedding[1:]
        prediction_error = _normalized_prediction_error(predicted, actual, test_embedding)

        complexity_penalty = complexity_weight * dimension
        total_score = reconstruction_error + prediction_error + complexity_penalty
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


def project_to_dimension_with_pca(states: list[list[float]], dimension: int) -> list[list[float]]:
    try:
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError("PCA method requires science dependencies: pip install -e '.[science]'") from error

    if dimension < 1:
        raise ValueError("dimension 必须 >= 1。")

    data = np.asarray(states, dtype=float)
    upper_dim = min(dimension, data.shape[1], data.shape[0])
    embedding = PCA(n_components=upper_dim, random_state=0).fit_transform(data)
    return embedding.tolist()


def _normalized_prediction_error(predicted, actual, embedding) -> float:
    import numpy as np

    if len(actual) == 0:
        return 1.0

    model_error = float(np.mean((predicted - actual) ** 2))
    baseline_steps = embedding[1:] - embedding[:-1]
    baseline_error = float(np.mean(baseline_steps ** 2)) or 1.0
    return model_error / baseline_error

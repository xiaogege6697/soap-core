"""Class-conditional geometry metrics for Neural Collapse analysis.

Implements NC1 ratio, centroid effective rank, Fisher separation,
and ETF deviation per v0.7.6 frozen spec.
"""

import numpy as np

from soap.apps.training.representation import effective_rank_from_covariance

# Small epsilon to prevent division by zero
EPS = 1e-8


def class_means(X: np.ndarray, labels: np.ndarray, K: int) -> tuple:
    """Compute per-class means and global mean.

    Args:
        X: (N, D) representation matrix.
        labels: (N,) integer class labels in [0, K-1].
        K: number of classes.

    Returns:
        M: (K, D) class centroid matrix (row c = mean of class c).
           Classes with no samples get zero vector.
        mu: (D,) global mean vector.
    """
    N, D = X.shape
    mu = X.mean(axis=0)  # (D,)

    M = np.zeros((K, D), dtype=X.dtype)
    for c in range(K):
        mask = labels == c
        if mask.any():
            M[c] = X[mask].mean(axis=0)

    return M, mu


def nc1_ratio(X: np.ndarray, labels: np.ndarray, K: int = 5) -> float:
    """NC1 ratio: within-class variance / total variance.

    Measures class-conditional collapse — lower values indicate
    samples are collapsing toward their class centroids.

    Args:
        X: (N, D) representation matrix.
        labels: (N,) integer class labels in [0, K-1].
        K: number of classes.

    Returns:
        NC1 ratio as float. 0 = perfect within-class collapse.
    """
    M, mu = class_means(X, labels, K)
    N = X.shape[0]

    # Within-class variance: Σ_c Σ_{x∈X_c} ||x − μ_c||² / N
    within_var = 0.0
    for c in range(K):
        mask = labels == c
        if mask.any():
            diffs = X[mask] - M[c]  # (n_c, D)
            within_var += np.sum(diffs * diffs)
    within_var /= N

    # Total variance: Σ_x ||x − μ||² / N
    total_diffs = X - mu  # (N, D)
    total_var = np.sum(total_diffs * total_diffs) / N

    return float(within_var / (total_var + EPS))


def centroid_effective_rank(X: np.ndarray, labels: np.ndarray, K: int = 5) -> float:
    """Effective rank of class centroid covariance matrix.

    Measures how many independent directions the class centroids span.
    Benign NC → K (full-rank separation); pathological → < K.

    Args:
        X: (N, D) representation matrix.
        labels: (N,) integer class labels in [0, K-1].
        K: number of classes.

    Returns:
        Effective rank of centroid covariance as float.
    """
    M, mu = class_means(X, labels, K)

    # Center centroids by global mean
    M_centered = M - mu  # (K, D)

    # Covariance of centroids: (D, D) from (K, D) matrix
    # np.cov with rowvar=False treats each row as an observation,
    # each column as a variable → cov is (D, D)
    cov_M = np.cov(M_centered, rowvar=False)

    return float(effective_rank_from_covariance(cov_M))


def fisher_separation(X: np.ndarray, labels: np.ndarray, K: int = 5) -> float:
    """Fisher separation: trace(S_b) / trace(S_w).

    Ratio of between-class scatter to within-class scatter.
    Benign NC → large (well-separated centroids, tight clusters).

    Args:
        X: (N, D) representation matrix.
        labels: (N,) integer class labels in [0, K-1].
        K: number of classes.

    Returns:
        Fisher separation as float.
    """
    M, mu = class_means(X, labels, K)

    # Between-class scatter S_b = Σ_c n_c (μ_c − μ)(μ_c − μ)^T
    # trace(S_b) = Σ_c n_c ||μ_c − μ||²
    trace_sb = 0.0
    for c in range(K):
        mask = labels == c
        n_c = mask.sum()
        if n_c > 0:
            diff = M[c] - mu  # (D,)
            trace_sb += n_c * np.dot(diff, diff)

    # Within-class scatter S_w = Σ_c Σ_{x∈X_c} (x − μ_c)(x − μ_c)^T
    # trace(S_w) = Σ_c Σ_{x∈X_c} ||x − μ_c||²
    trace_sw = 0.0
    for c in range(K):
        mask = labels == c
        if mask.any():
            diffs = X[mask] - M[c]  # (n_c, D)
            trace_sw += np.sum(diffs * diffs)

    return float(trace_sb / (trace_sw + EPS))


def etf_deviation(
    X: np.ndarray, labels: np.ndarray, K: int = 5
) -> float:
    """ETF deviation: class centroid geometry vs equiangular tight frame.

    Measures how far the class centroid inner-product matrix is from
    the ideal ETF simplex. Benign NC → 0.

    Args:
        X: (N, D) representation matrix.
        labels: (N,) integer class labels in [0, K-1].
        K: number of classes.

    Returns:
        ETF deviation as float. 0 = perfect ETF alignment.
    """
    M, mu = class_means(X, labels, K)

    # Row-normalize centroids (unit norm); zero vectors stay zero
    norms = np.linalg.norm(M, axis=1, keepdims=True)  # (K, 1)
    # Avoid division by zero for empty classes
    norms = np.where(norms < EPS, 1.0, norms)
    M_n = M / norms  # (K, D)

    # Inner product matrix G = M_n @ M_n^T  (K, K)
    G = M_n @ M_n.T

    # Ideal ETF matrix: diagonal 1, off-diagonal -1/(K-1)
    off_diag = -1.0 / (K - 1)
    G_etf = np.full((K, K), off_diag, dtype=X.dtype)
    np.fill_diagonal(G_etf, 1.0)

    # Frobenius norm deviation
    diff = G - G_etf
    num = np.sqrt(np.sum(diff * diff))
    den = np.sqrt(np.sum(G_etf * G_etf)) + EPS

    return float(num / den)

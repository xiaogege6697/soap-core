"""Unit tests for soap/apps/training/classcond_metrics.py."""

import numpy as np
import pytest
from soap.apps.training.classcond_metrics import (
    class_means,
    nc1_ratio,
    centroid_effective_rank,
    fisher_separation,
    etf_deviation,
)


# --------------- mock data helpers ---------------

def make_tight(K, D, n_per, sigma, rng=None):
    """每类样本紧密围绕各自的 class centroid（小噪声 sigma）。"""
    rng = np.random.default_rng(rng)
    # centroids: 前 K 个标准基方向，缩放以保证彼此分离
    centroids = np.eye(D)[:K] * 10.0  # [K, D]
    X, labels = [], []
    for k in range(K):
        # 每类 n_per 个样本 = centroid + 小高斯噪声
        samples = centroids[k] + rng.normal(0, sigma, size=(n_per, D))
        X.append(samples)
        labels.extend([k] * n_per)
    return np.vstack(X), np.array(labels)


def make_spread(K, D, n_per, sigma, rng=None):
    """每类样本散布较大（大噪声 sigma），centroid 与 make_tight 相同。"""
    return make_tight(K, D, n_per, sigma, rng=rng)


def make_collapsed(K, D, n_per, rng=None):
    """所有 class centroid 完全相同且样本无噪声（类均值严格相同 → cov(M)=0 → rank=0）。
    rng 保留参数兼容但不用（完全确定性坍缩）。
    """
    centroid = np.ones(D) * 5.0  # 所有类共享同一 centroid
    samples_one_class = np.tile(centroid, (n_per, 1))  # 每类 n_per 个完全相同的样本
    X = np.tile(samples_one_class, (K, 1))  # K 类，每类样本完全相同
    labels = np.repeat(np.arange(K), n_per)
    return X, labels


def _make_equiangular_centroids(K, D, scale=10.0):
    """构造 K 个等角向量（simplex ETF vertices），嵌入到 D 维空间。

    构造方法: K 维标准基去中心化、归一化，得到 pairwise dot = -1/(K-1)。
    """
    basis = np.eye(K)  # [K, K]
    centroid = basis.mean(axis=0, keepdims=True)  # [1, K]
    centered = basis - centroid  # [K, K]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / norms  # [K, K], 每行单位向量
    # 零填充嵌入到 D 维
    if D > K:
        unit = np.hstack([unit, np.zeros((K, D - K))])
    return unit * scale  # [K, D]


# --------------- 测试参数 ---------------

SEED = 42
K, D, N_PER = 5, 10, 80  # 5 类, 10 维, 每类 80 样本


# --------------- test cases ---------------

def test_nc1_ratio():
    """tight nc1 应显著小于 spread nc1（类内紧 → within/total 小）。"""
    X_t, y_t = make_tight(K, D, N_PER, sigma=0.1, rng=SEED)
    X_s, y_s = make_spread(K, D, N_PER, sigma=5.0, rng=SEED)

    nc1_tight = nc1_ratio(X_t, y_t, K=K)
    nc1_spread = nc1_ratio(X_s, y_s, K=K)

    assert nc1_tight < nc1_spread, (
        f"tight nc1={nc1_tight:.4f} should be < spread nc1={nc1_spread:.4f}"
    )


def test_centroid_effective_rank():
    """分离类中心 → eff rank 接近 K-1；坍缩 → 接近 1。"""
    # 分离情况: K=5 → rank 应接近 4，允许 [2.5, 4.5]
    X_sep, y_sep = make_tight(K, D, N_PER, sigma=0.5, rng=SEED)
    rank_sep = centroid_effective_rank(X_sep, y_sep, K=K)
    assert 2.5 <= rank_sep <= 4.5, (
        f"separated eff rank={rank_sep:.2f} should be in [2.5, 4.5]"
    )

    # 坍缩情况: 所有中心相同 → rank 接近 1
    X_col, y_col = make_collapsed(K, D, N_PER, rng=SEED)
    rank_col = centroid_effective_rank(X_col, y_col, K=K)
    assert rank_col < 2.0, (
        f"collapsed eff rank={rank_col:.2f} should be < 2.0"
    )


def test_fisher_separation():
    """tight Fisher > spread Fisher；collapsed Fisher 很小。"""
    X_t, y_t = make_tight(K, D, N_PER, sigma=0.1, rng=SEED)
    X_s, y_s = make_spread(K, D, N_PER, sigma=5.0, rng=SEED)
    X_c, y_c = make_collapsed(K, D, N_PER, rng=SEED)

    fish_tight = fisher_separation(X_t, y_t, K=K)
    fish_spread = fisher_separation(X_s, y_s, K=K)
    fish_coll = fisher_separation(X_c, y_c, K=K)

    # 紧类 trace(Sb)/trace(Sw) > 松类
    assert fish_tight > fish_spread, (
        f"tight Fisher={fish_tight:.4f} should be > spread Fisher={fish_spread:.4f}"
    )
    # 坍缩时类间方差极小 → Fisher 很小
    assert fish_coll < fish_spread, (
        f"collapsed Fisher={fish_coll:.4f} should be < spread Fisher={fish_spread:.4f}"
    )


def test_etf_deviation():
    """等角类中心 → ETF deviation 小；随机中心 → deviation 较大。"""
    rng = np.random.default_rng(SEED)

    # 等角 centroids 数据: 构造 ETF simplex + 极小噪声
    centroids_etf = _make_equiangular_centroids(K, D, scale=10.0)
    X_etf, y_etf = [], []
    for k in range(K):
        X_etf.append(centroids_etf[k] + rng.normal(0, 0.05, size=(N_PER, D)))
        y_etf.extend([k] * N_PER)
    X_etf = np.vstack(X_etf)
    y_etf = np.array(y_etf)

    dev_etf = etf_deviation(X_etf, y_etf, K=K)

    # 随机 centroids 数据: 各中心随机高斯采样
    rng2 = np.random.default_rng(123)
    centroids_rand = rng2.normal(size=(K, D)) * 10.0
    X_rand, y_rand = [], []
    for k in range(K):
        X_rand.append(centroids_rand[k] + rng2.normal(0, 0.05, size=(N_PER, D)))
        y_rand.extend([k] * N_PER)
    X_rand = np.vstack(X_rand)
    y_rand = np.array(y_rand)

    dev_rand = etf_deviation(X_rand, y_rand, K=K)

    # 等角 deviation 应显著小于随机
    assert dev_etf < dev_rand, (
        f"ETF deviation={dev_etf:.4f} should be < random deviation={dev_rand:.4f}"
    )


def test_class_means():
    """class_means: M 形状 (K, D)，每行是对应 class 均值；mu 是全局均值。"""
    X, y = make_tight(K, D, N_PER, sigma=0.5, rng=SEED)
    M, mu = class_means(X, y, K=K)

    # M 形状检查
    assert M.shape == (K, D), f"M shape {M.shape} != expected ({K}, {D})"
    # mu 形状检查
    assert mu.shape == (D,), f"mu shape {mu.shape} != expected ({D},)"

    # mu 应等于全局样本均值
    np.testing.assert_allclose(
        mu, X.mean(axis=0), atol=1e-10,
        err_msg="mu should equal global mean of X",
    )

    # M 每行应等于对应 class 的样本均值
    for k in range(K):
        expected_mean = X[y == k].mean(axis=0)
        np.testing.assert_allclose(
            M[k], expected_mean, atol=1e-10,
            err_msg=f"M[{k}] should equal class {k} mean",
        )

"""
SOAP-Core v0.7 表示几何指标模块。

包含用于分析神经网络激活表示的几何特性函数。
"""
import numpy as np


def activation_covariance(activations):
    """activations: (n_samples, n_features) numpy array 或 torch tensor。返回协方差矩阵 (n_features, n_features)。"""
    # 若为 torch tensor，先转 numpy
    if hasattr(activations, 'detach'):
        activations = activations.detach().cpu().numpy()
    A = np.asarray(activations, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    cov = np.cov(A, rowvar=False)
    # 若 n_features==1，np.cov 返回标量，需 reshape
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return cov


def effective_rank_from_covariance(cov):
    """entropy effective rank: exp(-sum(p_i log p_i))，p_i = lambda_i / sum(lambda)。"""
    cov = np.asarray(cov, dtype=float)
    eigvals = np.linalg.eigvalsh(cov)  # 升序特征值
    eigvals = np.maximum(eigvals, 0.0)  # 数值安全，clip 负
    s = eigvals.sum()
    if s <= 0:
        return 0.0
    p = eigvals / s
    p = p[p > 0]  # 避免 log(0)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def representation_variance(activations):
    """激活总方差（所有元素的方差）。"""
    if hasattr(activations, 'detach'):
        activations = activations.detach().cpu().numpy()
    return float(np.var(np.asarray(activations, dtype=float)))


def collapse_score(activations):
    """collapse proxy = 1 / effective_rank。eff_rank 越低（表示越坍缩）score 越高。注意这是 proxy。"""
    eff = effective_rank_from_covariance(activation_covariance(activations))
    if eff <= 0:
        return 1e6
    return float(1.0 / eff)


if __name__ == "__main__":
    # self-test：造随机 (200, 16) 激活，打印四函数值；再造坍缩激活（全部相同）看 effective_rank 趋近 1、collapse_score 趋近 1
    rng = np.random.RandomState(0)
    A = rng.randn(200, 16)
    print("random activations:")
    print("  effective_rank =", effective_rank_from_covariance(activation_covariance(A)))
    print("  representation_variance =", representation_variance(A))
    print("  collapse_score =", collapse_score(A))
    A_collapsed = np.ones((200, 16)) * 3.0  # 完全坍缩（秩1）
    print("collapsed activations:")
    print("  effective_rank =", effective_rank_from_covariance(activation_covariance(A_collapsed)))
    print("  collapse_score =", collapse_score(A_collapsed))

"""
Algorithm 1 (paper Sec. 3.3.2): Sinkhorn-OT interpolation between two domains'
per-class CLIP embedding distributions, via barycentric projection.

Given X (N_i, d) from domain i, class c and Y (N_j, d) from domain j, class c:
  1. Cost matrix M_{ab} = ||x_a - y_b||^2
  2. Uniform marginals a = 1/N_i, b = 1/N_j
  3. Entropic Sinkhorn (eps=0.01) -> transport plan T* (N_i, N_j)
  4. Barycentric projection: ybar_a = sum_b T*_{ab} y_b / sum_b T*_{ab}
  5. Interpolants: z_a(t) = (1-t) x_a + t * ybar_a

Geometry fix (see check_transport_degeneracy.py diagnostic): the paper's Sec.
3.6/Supp. S1 literally says this operates on unnormalized CLIP embeddings, but
raw embeddings here have norm~20-21, giving squared-distance costs of ~165-788.
At that scale, eps=0.01 is negligible relative to the cost matrix, so Sinkhorn
collapses to a near-hard nearest-neighbor assignment (measured: row entropy only
4.3% of max possible, effective support ~1.4 points out of ~380, 86% of mass on
the single top match) - not the smooth, distributional barycentric projection
Algorithm 1 is meant to compute. This is also independently consistent with the
paper's own Supp. S1 entropic-bias calculation (D^2_max~0.81), which only makes
sense at a normalized scale.

Fix: compute the actual transport plan and barycentric projection on L2-NORMALIZED
embeddings (where eps=0.01 is a meaningful regularizer - measured: row entropy 29%
of max, effective support ~7 points, 48% top-match mass), then rescale the
resulting interpolant back to a raw-embedding norm scale (linearly interpolated
between the source and transport-weighted target raw norms, consistent with
Theorem 3.4's linear-in-t moment variation) before returning it - since the unCLIP
generator needs raw-scale conditioning vectors (validated by the Stage 0 smoke
test's cosine-similarity gate).
"""
import numpy as np
import ot


def sinkhorn_barycentric_interpolants(X: np.ndarray, Y: np.ndarray, t: float,
                                       eps: float = 0.01, stop_thr: float = 1e-6,
                                       num_iter_max: int = 2000):
    """Returns z (N_i, d): one interpolant per row of X, at interpolation weight t
    towards X's barycentric projection into Y's support. OT geometry (cost matrix,
    Sinkhorn, barycentric projection) is computed on L2-normalized embeddings; the
    result is rescaled to a raw-embedding norm scale for downstream use (generation
    conditioning, eps_extrap grid-point construction)."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    n_i, n_j = X.shape[0], Y.shape[0]

    x_norms = np.linalg.norm(X, axis=1, keepdims=True)
    y_norms = np.linalg.norm(Y, axis=1, keepdims=True)
    Xn = X / np.clip(x_norms, 1e-12, None)
    Yn = Y / np.clip(y_norms, 1e-12, None)

    a = np.full(n_i, 1.0 / n_i)
    b = np.full(n_j, 1.0 / n_j)

    M = ot.dist(Xn, Yn, metric="sqeuclidean")
    try:
        T = ot.sinkhorn(a, b, M, reg=eps, stopThr=stop_thr, numItermax=num_iter_max, method="sinkhorn_log")
    except Exception:
        T = ot.sinkhorn(a, b, M, reg=eps, stopThr=stop_thr, numItermax=num_iter_max)
    if not np.all(np.isfinite(T)):
        raise RuntimeError("Sinkhorn transport plan contains non-finite entries even in log-domain mode.")

    row_mass = T.sum(axis=1, keepdims=True)
    row_mass = np.clip(row_mass, 1e-12, None)

    ybar_n = (T @ Yn) / row_mass                    # (N_i, d) barycentric direction in Y-space
    ybar_raw_norm = (T @ y_norms) / row_mass          # (N_i, 1) transport-weighted target raw norm

    z_n = (1 - t) * Xn + t * ybar_n                   # interpolated direction (not unit norm in general)
    z_n_norm = np.linalg.norm(z_n, axis=1, keepdims=True)
    z_direction = z_n / np.clip(z_n_norm, 1e-12, None)

    target_raw_norm = (1 - t) * x_norms + t * ybar_raw_norm  # linear-in-t raw norm (Theorem 3.4)
    z = z_direction * target_raw_norm

    return z.astype(np.float32)

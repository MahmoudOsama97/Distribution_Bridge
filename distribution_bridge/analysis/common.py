"""Shared utilities for Stage 4 analysis (E3 eps_gen, E4 eps_extrap)."""
import itertools
import os

import numpy as np
import ot

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def load_embeddings(embeddings_dir, domain, cls):
    data = np.load(os.path.join(embeddings_dir, f"{domain}_{cls}.npz"))
    return data["embeddings"].astype(np.float64)  # (N, 1024)


def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


# NOTE on normalization: the paper states TWICE, explicitly, that OT (the main
# method), LERP, and Gaussian-W2 all operate in UNNORMALIZED R^d - only the SLERP
# baseline uses L2-normalized prototypes (Sec. 3.6: "LERP, Gaussian-W2, and OT
# operate in unnormalized R^d"; Supp. S1: "CLIP embeddings, OT cost matrices, and
# stored statistics ... are computed in unnormalized R^d"). Per author decision,
# Stage 4 analysis below follows that stated convention and stays unnormalized -
# even though this produces a measured PACS D_max (~24.5) that is ~24x the paper's
# originally reported Table 7 value (~1.02) and Supp. S1's D^2_max~0.81 reference.
# A direct empirical check (jobs/check_embedding_scale.sh) confirmed our raw
# embeddings are numerically identical (cosine sim 0.99999) to native open_clip
# ViT-H-14/laion2b_s32b_b79k output, and that L2-normalizing them WOULD bring the
# distance into the paper's originally reported range (~0.877 vs ~0.81-1.08) - but
# that would contradict the paper's own explicit unnormalized-R^d statement, so this
# is flagged as a discrepancy for the response letter / errata rather than silently
# renormalized to match the old numbers.
NORMALIZE_FOR_ANALYSIS = False


def exact_w2(X, Y, normalize=NORMALIZE_FOR_ANALYSIS):
    """Exact discrete W2 distance (uniform marginals) between two point clouds,
    via POT's exact LP solver (ot.emd2), NOT entropic Sinkhorn."""
    if normalize:
        X = l2_normalize_rows(X)
        Y = l2_normalize_rows(Y)
    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    M = ot.dist(X, Y, metric="sqeuclidean")
    cost = ot.emd2(a, b, M)
    return float(np.sqrt(max(cost, 0.0)))


def compute_d_max(embeddings_dir, domains=DOMAINS, classes=CLASSES, normalize=NORMALIZE_FOR_ANALYSIS):
    """D_max = max_{i,j} W2(P_i, P_j): the maximum exact-W2 distance between any two
    source domains' FULL per-class empirical CLIP-embedding distributions (paper
    Prop 3.6 / Table 7 methodology - not a mean-embedding proxy). Used to normalize
    eps_gen / eps_extrap into a domain-shift-relative scale. See NORMALIZE_FOR_ANALYSIS
    note above for why this defaults to L2-normalized embeddings."""
    max_dist = 0.0
    for c in classes:
        for d1, d2 in itertools.combinations(domains, 2):
            X = load_embeddings(embeddings_dir, d1, c)
            Y = load_embeddings(embeddings_dir, d2, c)
            dist = exact_w2(X, Y, normalize=normalize)
            max_dist = max(max_dist, dist)
    return float(max_dist)

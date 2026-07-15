"""
N3 - eps_extrap (Definition 3.4) per held-out PACS target, l2-normalized
convention, exact OT for every reported distance.

Prerequisite: B4 Task 1's convention gate must have printed CONVENTION CONFIRMED
(results/sanity_check.txt). Standalone mode: if that file is missing or did not
confirm, this script recomputes the gate itself via n2_reconcile.task1_sanity_gate.

For each leave-one-domain-out target T and each class c:
  1. Build the 1-skeleton: for each of the 3 source pairs, run Algorithm 1
     (Sinkhorn eps=0.01 on l2-NORMALIZED clouds, barycentric projection) at
     t in {0.00, 0.05, ..., 1.00} (21 points). Interpolants are NOT rescaled or
     re-normalized after construction - they leave the unit sphere, which is
     expected (the skeleton lives in R^d).
  2. Exact W2 (ot.emd2, sqrt) from the normalized target class cloud to every one
     of the 3x21=63 interpolant clouds; take the min -> d_c(T).
  3. eps_extrap(T) = max_c d_c(T) (Definition 3.4); also report mean-over-classes.
  4. Normalize by B4's D_max.

No generation, no training. Writes only under --out_dir (kept separate from the
existing raw-convention results/eps_extrap_per_class.csv etc. to avoid
overwriting already-published PACS results computed under a different
convention).
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np
import ot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from distribution_bridge.analysis.common import DOMAINS, CLASSES, load_embeddings  # noqa: E402
from distribution_bridge.analysis.table7_reconcile import task1_sanity_gate  # noqa: E402

T_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)  # 21 points, {0.00, ..., 1.00}
SUBSAMPLE_N = 2000
SEED = 12345

# Coverage gains from the volume-matched K=1 -> K=5 ablation (PACS, ERM avg acc, pp).
COVERAGE_GAIN = {"art_painting": 0.73, "cartoon": 0.36, "photo": -1.30, "sketch": 3.48}


def l2_normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)


def maybe_subsample(X, rng, tag, subsampled_flags):
    if X.shape[0] > SUBSAMPLE_N:
        idx = rng.choice(X.shape[0], SUBSAMPLE_N, replace=False)
        subsampled_flags.append(tag)
        return X[idx]
    return X


def interpolants_normalized(Xn, Yn, t, eps=0.01, stop_thr=1e-6, num_iter_max=2000):
    """Algorithm 1, entirely in l2-normalized space. Xn, Yn already normalized.
    Returns z (N_i, d): interpolants NOT renormalized / NOT rescaled to raw norm -
    per Task 3 spec, the skeleton lives in R^d and interpolants may leave S^{d-1}."""
    n_i, n_j = Xn.shape[0], Yn.shape[0]
    a = np.full(n_i, 1.0 / n_i)
    b = np.full(n_j, 1.0 / n_j)
    M = ot.dist(Xn, Yn, metric="sqeuclidean")
    try:
        T = ot.sinkhorn(a, b, M, reg=eps, stopThr=stop_thr, numItermax=num_iter_max, method="sinkhorn_log")
    except Exception:
        T = ot.sinkhorn(a, b, M, reg=eps, stopThr=stop_thr, numItermax=num_iter_max)
    if not np.all(np.isfinite(T)):
        raise RuntimeError("Sinkhorn transport plan contains non-finite entries even in log-domain mode.")
    row_mass = np.clip(T.sum(axis=1, keepdims=True), 1e-12, None)
    ybar_n = (T @ Yn) / row_mass
    z = (1 - t) * Xn + t * ybar_n
    return z


def exact_w2_raw(X, Y):
    """Exact discrete W2 (ot.emd2, uniform marginals), no internal normalization -
    both inputs are used exactly as given."""
    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    M = ot.dist(X, Y, metric="sqeuclidean")
    cost = ot.emd2(a, b, M)
    return float(np.sqrt(max(cost, 0.0)))


def get_d_max(embeddings_dir, out_dir, results_dir_for_gate):
    """Reuse B4 Task 1's D_max if results/sanity_check.txt confirms the convention;
    otherwise recompute (standalone mode)."""
    sanity_path = os.path.join(results_dir_for_gate, "sanity_check.txt")
    if os.path.exists(sanity_path):
        with open(sanity_path) as f:
            text = f.read()
        if "CONVENTION CONFIRMED" in text:
            for line in text.splitlines():
                if line.startswith("D_max"):
                    d_max = float(line.split("=")[1].strip())
                    print(f"Reusing B4 Task 1's confirmed D_max = {d_max:.4f} (from {sanity_path})")
                    return d_max
        print("results/sanity_check.txt exists but did not confirm the convention - "
              "recomputing gate in standalone mode.")
    else:
        print(f"{sanity_path} not found - recomputing Task 1 gate in standalone mode.")
    gate_pass, mean, std, d_max, cells = task1_sanity_gate(embeddings_dir, out_dir)
    if not gate_pass:
        raise RuntimeError(
            f"CONVENTION MISMATCH: mean pairwise W2 = {mean:.4f}, outside [0.85, 1.20]. "
            "Stopping per the prerequisite gate - do not proceed to eps_extrap in an "
            "unconfirmed convention."
        )
    return d_max


def print_inventory(embeddings_dir, d_max):
    print("=== Inventory ===")
    for d in DOMAINS:
        counts = {}
        for c in CLASSES:
            f = os.path.join(embeddings_dir, f"{d}_{c}.npz")
            counts[c] = np.load(f)["embeddings"].shape[0] if os.path.exists(f) else None
        print(f"  {d}: {counts}")
    print(f"D_max in use: {d_max:.4f}")
    print(f"Grid: {len(T_GRID)} points, {T_GRID[0]}..{T_GRID[-1]}")
    print("=== End inventory ===\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--results_dir_for_gate", default=None,
                     help="where to look for/write results/sanity_check.txt (defaults to --out_dir's parent)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_dir_for_gate = args.results_dir_for_gate or os.path.dirname(args.out_dir.rstrip("/"))

    d_max = get_d_max(args.embeddings_dir, args.out_dir, results_dir_for_gate)
    print_inventory(args.embeddings_dir, d_max)

    rng = np.random.default_rng(SEED)
    subsampled_flags = []

    # Cache normalized embeddings + per-(pair,t) interpolant grids so we don't
    # rebuild the same skeleton once per class-target loop.
    norm_cache = {}

    def get_norm(domain, cls):
        key = (domain, cls)
        if key not in norm_cache:
            X = load_embeddings(args.embeddings_dir, domain, cls)
            X = maybe_subsample(X, rng, f"{domain}/{cls}", subsampled_flags)
            norm_cache[key] = l2_normalize_rows(X)
        return norm_cache[key]

    per_class_rows = []
    per_target_summary = []
    argmin_t_counter = {}

    for target in DOMAINS:
        sources = [d for d in DOMAINS if d != target]
        pairs = list(itertools.combinations(sources, 2))

        class_eps = {}
        for cls in CLASSES:
            target_n = get_norm(target, cls)

            best_dist = np.inf
            best_pair, best_t = None, None
            for domA, domB in pairs:
                Xn = get_norm(domA, cls)
                Yn = get_norm(domB, cls)
                for t in T_GRID:
                    z = interpolants_normalized(Xn, Yn, float(t))
                    dist = exact_w2_raw(target_n, z)
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = f"{domA}_{domB}"
                        best_t = float(t)

            class_eps[cls] = best_dist
            per_class_rows.append({
                "target": target, "cls": cls, "nearest_pair": best_pair, "argmin_t": best_t,
                "d_c": best_dist, "d_c_over_dmax": best_dist / d_max,
            })
            argmin_t_counter[best_t] = argmin_t_counter.get(best_t, 0) + 1
            print(f"  target={target} class={cls}: d_c={best_dist:.4f} "
                  f"({best_dist / d_max:.4f} D_max), nearest_pair={best_pair}, argmin_t={best_t}")

        vals = np.array(list(class_eps.values()))
        eps_max, eps_mean = float(vals.max()), float(vals.mean())
        eps_min_class = float(vals.min())
        per_target_summary.append({
            "target": target,
            "eps_extrap_max": eps_max, "eps_extrap_max_norm": eps_max / d_max,
            "eps_extrap_mean": eps_mean, "eps_extrap_mean_norm": eps_mean / d_max,
            "range_min_norm": eps_min_class / d_max, "range_max_norm": eps_max / d_max,
            "coverage_gain": COVERAGE_GAIN.get(target),
        })
        print(f"target={target}: eps_extrap(T)/D_max [max]={eps_max / d_max:.4f}, "
              f"[mean]={eps_mean / d_max:.4f}, range=[{eps_min_class / d_max:.4f}, {eps_max / d_max:.4f}]\n")

    # --- outputs ---
    class_csv = os.path.join(args.out_dir, "eps_extrap_per_class.csv")
    with open(class_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "cls", "nearest_pair", "argmin_t", "d_c", "d_c_over_dmax"])
        w.writeheader()
        w.writerows(per_class_rows)
    print(f"Wrote {class_csv}")

    def target_order_key(row):
        # display order matching the spec's table: photo, art_painting, cartoon, sketch
        order = {"photo": 0, "art_painting": 1, "cartoon": 2, "sketch": 3}
        return order.get(row["target"], 99)

    ordered = sorted(per_target_summary, key=target_order_key)

    md_lines = [
        "| Held-out target | eps_extrap/D_max (max over classes, Def. 3.4) | mean over classes | "
        "Per-class range | Coverage gain (K=1->K=5 matched, pp) |",
        "|---|---|---|---|---|",
    ]
    for row in ordered:
        md_lines.append(
            f"| {row['target']} | {row['eps_extrap_max_norm']:.4f} | {row['eps_extrap_mean_norm']:.4f} | "
            f"[{row['range_min_norm']:.4f}, {row['range_max_norm']:.4f}] | {row['coverage_gain']:+.2f} |"
        )
    md_path = os.path.join(args.out_dir, "eps_extrap.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-hoc extrapolation distance (Definition 3.4) per held-out PACS target, "
        r"measured on $\ell_2$-normalized embeddings with exact optimal transport; the 1-skeleton "
        r"is discretized at 21 points per source-pair geodesic via Algorithm 1. Coverage gains from "
        r"the volume-matched $K$ ablation are shown for reference.}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Target & $\epsilon_{extrap}/D_{max}$ (max) & mean & range & Coverage gain (pp) \\",
        r"\midrule",
    ]
    for row in ordered:
        tex_lines.append(
            f"{row['target']} & {row['eps_extrap_max_norm']:.4f} & {row['eps_extrap_mean_norm']:.4f} & "
            f"[{row['range_min_norm']:.4f}, {row['range_max_norm']:.4f}] & {row['coverage_gain']:+.2f} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(args.out_dir, "eps_extrap.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Wrote {tex_path}")

    # Spearman correlation (illustrative only, n=4)
    from scipy.stats import spearmanr
    eps_vals = [row["eps_extrap_max_norm"] for row in ordered]
    gain_vals = [row["coverage_gain"] for row in ordered]
    sr, sp = spearmanr(eps_vals, gain_vals)

    ranking = sorted(per_target_summary, key=lambda r: -r["eps_extrap_max_norm"])
    ranking_str = " > ".join(f"{r['target']} ({r['eps_extrap_max_norm']:.4f})" for r in ranking)

    mean_ranking = sorted(per_target_summary, key=lambda r: -r["eps_extrap_mean_norm"])
    mean_ranking_str = " > ".join(f"{r['target']} ({r['eps_extrap_mean_norm']:.4f})" for r in mean_ranking)
    ranks_preserved = [r["target"] for r in ranking] == [r["target"] for r in mean_ranking]

    farthest = ranking[0]["target"]
    farthest_gain = COVERAGE_GAIN[farthest]
    largest_gain_target = max(COVERAGE_GAIN, key=COVERAGE_GAIN.get)
    aligns = farthest == largest_gain_target

    argmin_dist_str = ", ".join(f"t={t}: {n}" for t, n in sorted(argmin_t_counter.items()))

    verdict = (
        f"The farthest target by max-over-classes eps_extrap is **{farthest}** "
        f"(gain {farthest_gain:+.2f}pp), while the target with the largest volume-matched "
        f"coverage gain is **{largest_gain_target}** ({COVERAGE_GAIN[largest_gain_target]:+.2f}pp). "
        f"{'These agree' if aligns else 'These do NOT agree'} - "
        f"{'the farthest target from the source 1-skeleton is also the one that benefited most from denser geodesic coverage, consistent with the covering-radius term being a meaningful part of the risk gap even for this off-skeleton target.' if aligns else 'the farthest target from the 1-skeleton is not the one that gained most from coverage, so this measurement does not support the hoped-for pattern on PACS as directly as expected - reported as measured, not adjusted.'} "
        f"Spearman rho(eps_extrap_max, coverage_gain) = {sr:.3f} (p={sp:.3f}, n=4, illustrative only - "
        f"not statistically inferential at this sample size). "
        f"The mean-over-classes variant {'preserves' if ranks_preserved else 'does NOT preserve'} "
        f"the max-over-classes target ranking."
    )

    summary_lines = [
        "# N3 summary - eps_extrap (Definition 3.4), l2-normalized convention",
        "",
        f"D_max used: {d_max:.4f} (from B4 Task 1's confirmed gate, or recomputed standalone if that was missing)",
        f"Subsampling triggered: {bool(subsampled_flags)} (seed={SEED}, threshold={SUBSAMPLE_N} pts/cloud)"
        + (f" -- cells: {subsampled_flags}" if subsampled_flags else ""),
        "",
        "## Ranking (max over classes)",
        ranking_str,
        "",
        "## Ranking (mean over classes)",
        mean_ranking_str,
        "",
        "## argmin-t distribution (which geodesic position was closest, across all 28 target-class cells)",
        argmin_dist_str,
        "",
        "## Verdict",
        verdict,
        "",
        "## Main table",
        "",
    ] + md_lines

    summary_path = os.path.join(args.out_dir, "eps_extrap_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Wrote {summary_path}")
    print("\n" + "\n".join(summary_lines))
    print("\n=== N3 (eps_extrap) complete ===")


if __name__ == "__main__":
    main()

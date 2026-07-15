"""
Stage 4 / E4 - eps_extrap analysis (answers R1.1 + R1.2 + AE). Pure embedding math
on Stage 1 caches - no synthetic data, no training required.

For each leave-one-domain-out config (target T, sources = the other 3 domains) and
each class c:
  1. Discretize each of the 3 source-pair geodesics with Algorithm 1 (Sinkhorn
     barycentric interpolation, reusing ot_interpolate.py - the same construction
     Stage 2a uses) at a fine grid t in {0.00, 0.05, ..., 1.00}.
  2. At each grid point, compute the exact discrete W2 distance (POT's ot.emd2,
     NOT Sinkhorn, to avoid entropic bias in the *reported* distance) between the
     held-out target domain's raw embedding cloud P_hat{T,c} and the grid point's
     interpolant cloud.
  3. Take the min over the 1-skeleton (all pairs x all grid points) -> eps_extrap(T,c).
  4. eps_extrap(T) = max over classes; also report the mean-over-classes variant.
  5. Normalize by D_max.

Correlation against Stage-3 E1 per-target accuracy deltas is a separate, optional
step (--accuracy_deltas_csv) run once those results exist.

Usage:
  python eps_extrap.py --embeddings_dir .../results/stage1_embeddings --out_dir .../results
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from distribution_bridge.generation.ot_interpolate import sinkhorn_barycentric_interpolants  # noqa: E402
from distribution_bridge.analysis.common import DOMAINS as PACS_DOMAINS, CLASSES as PACS_CLASSES  # noqa: E402
from distribution_bridge.analysis.common import load_embeddings, compute_d_max, exact_w2  # noqa: E402

T_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)  # {0.00, 0.05, ..., 1.00}


def discover_domains_classes(embeddings_dir, domains_override):
    """Recover (domains, classes) from cached {domain}_{class}.npz filenames, given
    an explicit domain list (needed since domain/class names may contain '_' or
    spaces, e.g. OfficeHome's "Real World" domain and "Alarm_Clock" class - can't
    split the filename unambiguously without knowing the domain names up front)."""
    classes = set()
    for fname in os.listdir(embeddings_dir):
        if not fname.endswith(".npz"):
            continue
        for d in domains_override:
            prefix = d + "_"
            if fname.startswith(prefix):
                classes.add(fname[len(prefix):-len(".npz")])
                break
    return sorted(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--accuracy_deltas_csv", default=None,
                     help="optional CSV with columns target,accuracy_delta for post-hoc correlation")
    ap.add_argument("--domains", default=None, help="comma-separated; default: PACS's 4 domains")
    ap.add_argument("--classes", default=None,
                     help="comma-separated; default: PACS's 7 classes, or auto-discovered from "
                          "--embeddings_dir filenames if --domains is given without --classes")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.domains:
        domains = args.domains.split(",")
        classes = args.classes.split(",") if args.classes else discover_domains_classes(args.embeddings_dir, domains)
    else:
        domains, classes = PACS_DOMAINS, PACS_CLASSES
    print(f"Domains ({len(domains)}): {domains}")
    print(f"Classes ({len(classes)}): {classes}")

    print("Computing D_max from Stage 1 embeddings...")
    d_max = compute_d_max(args.embeddings_dir, domains=domains, classes=classes)
    print(f"D_max = {d_max:.4f}")

    per_target_class_rows = []
    per_target_summary = []

    for target in domains:
        sources = [d for d in domains if d != target]
        pairs = list(itertools.combinations(sources, 2))

        class_eps = {}
        for cls in classes:
            target_cloud = load_embeddings(args.embeddings_dir, target, cls)
            min_dist_this_class = np.inf

            for domA, domB in pairs:
                X = load_embeddings(args.embeddings_dir, domA, cls)
                Y = load_embeddings(args.embeddings_dir, domB, cls)
                for t in T_GRID:
                    grid_cloud = sinkhorn_barycentric_interpolants(X, Y, float(t)).astype(np.float64)
                    dist = exact_w2(target_cloud, grid_cloud)
                    if dist < min_dist_this_class:
                        min_dist_this_class = dist

            class_eps[cls] = min_dist_this_class
            per_target_class_rows.append({
                "target": target, "cls": cls,
                "eps_extrap": min_dist_this_class, "eps_extrap_norm": min_dist_this_class / d_max,
            })
            print(f"  target={target} class={cls}: eps_extrap={min_dist_this_class:.4f} "
                  f"({min_dist_this_class / d_max:.4f} D_max)")

        vals = np.array(list(class_eps.values()))
        eps_max = float(vals.max())
        eps_mean = float(vals.mean())
        per_target_summary.append({
            "target": target,
            "eps_extrap_max": eps_max, "eps_extrap_max_norm": eps_max / d_max,
            "eps_extrap_mean": eps_mean, "eps_extrap_mean_norm": eps_mean / d_max,
        })
        print(f"target={target}: eps_extrap(T) [max over classes] = {eps_max:.4f} "
              f"({eps_max / d_max:.4f} D_max); mean-over-classes = {eps_mean:.4f} ({eps_mean / d_max:.4f} D_max)")

    class_csv = os.path.join(args.out_dir, "eps_extrap_per_class.csv")
    with open(class_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "cls", "eps_extrap", "eps_extrap_norm"])
        writer.writeheader()
        writer.writerows(per_target_class_rows)
    print(f"Wrote per-class eps_extrap -> {class_csv}")

    summary_csv = os.path.join(args.out_dir, "eps_extrap_per_target.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "target", "eps_extrap_max", "eps_extrap_max_norm", "eps_extrap_mean", "eps_extrap_mean_norm"])
        writer.writeheader()
        writer.writerows(per_target_summary)
    print(f"Wrote per-target eps_extrap summary -> {summary_csv}")

    tex_path = os.path.join(args.out_dir, "table_eps_extrap.tex")
    with open(tex_path, "w") as f:
        f.write(r"\begin{tabular}{lcc}" + "\n" + r"\toprule" + "\n")
        f.write(r"Target & $\epsilon_{extrap}(T)/D_{max}$ (max) & $\epsilon_{extrap}(T)/D_{max}$ (mean) \\" + "\n")
        f.write(r"\midrule" + "\n")
        for row in per_target_summary:
            f.write(f"{row['target']} & {row['eps_extrap_max_norm']:.4f} & {row['eps_extrap_mean_norm']:.4f} \\\\\n")
        f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    print(f"Wrote table -> {tex_path}")

    if args.accuracy_deltas_csv and os.path.exists(args.accuracy_deltas_csv):
        from scipy.stats import pearsonr, spearmanr
        deltas = {}
        with open(args.accuracy_deltas_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                deltas[row["target"]] = float(row["accuracy_delta"])

        eps_vals, delta_vals = [], []
        for row in per_target_summary:
            if row["target"] in deltas:
                eps_vals.append(row["eps_extrap_max_norm"])
                delta_vals.append(deltas[row["target"]])

        if len(eps_vals) >= 3:
            pr, pp = pearsonr(eps_vals, delta_vals)
            sr, sp = spearmanr(eps_vals, delta_vals)
            print(f"\nCorrelation eps_extrap(T) vs DDB accuracy gain: "
                  f"Pearson r={pr:.3f} (p={pp:.3f}), Spearman rho={sr:.3f} (p={sp:.3f})")
        else:
            print("\nNot enough matched (target, delta) pairs for correlation.")
    else:
        print("\nNo --accuracy_deltas_csv provided (or file not found) - "
              "run this again once Stage 3 E1 results are collected to get the correlation.")


if __name__ == "__main__":
    main()

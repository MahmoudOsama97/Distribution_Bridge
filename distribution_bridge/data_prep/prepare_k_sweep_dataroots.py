"""
Build K=1 and K=3 data roots for E2 (K sweep, R1.5), reusing subsets of the
Stage 2a OT weights (no new generation needed, per the revision plan):

  K=1: t in {3/6}              -> data_k1/pacs/  (1 weight x 6 pairs = 6 synthetic domains)
  K=3: t in {1/6, 3/6, 5/6}    -> data_k3/pacs/  (3 weights x 6 pairs = 18 synthetic domains)
  K=5: all weights             -> already exists as data_ot/ (reuse E1 OT results directly)

Usage:
  python prepare_k_sweep_dataroots.py --base_data_root .../data \
      --k1_root .../data_k1 --k3_root .../data_k3
"""
import argparse
import os

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]


def build_root(base_pacs_dir, out_root, weight_tags):
    out_pacs = os.path.join(out_root, "pacs")
    os.makedirs(out_pacs, exist_ok=True)

    for d in DOMAINS:
        link_path = os.path.join(out_pacs, d)
        target = os.path.join(base_pacs_dir, d)
        if not os.path.exists(link_path):
            os.symlink(target, link_path)

    n_syn = 0
    for folder in sorted(os.listdir(base_pacs_dir)):
        if not folder.startswith("SynDomain_OT_"):
            continue
        if any(folder.endswith(f"_t{tag}") for tag in weight_tags):
            link_path = os.path.join(out_pacs, folder)
            target = os.path.join(base_pacs_dir, folder)
            if not os.path.exists(link_path):
                os.symlink(target, link_path)
            n_syn += 1
    print(f"{out_root}: linked 4 original domains + {n_syn} OT synthetic domains (weights {weight_tags})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data_root", required=True)
    ap.add_argument("--k1_root", required=True)
    ap.add_argument("--k3_root", required=True)
    args = ap.parse_args()

    args.base_data_root = os.path.abspath(args.base_data_root)
    args.k1_root = os.path.abspath(args.k1_root)
    args.k3_root = os.path.abspath(args.k3_root)

    base_pacs_dir = os.path.join(args.base_data_root, "pacs")
    build_root(base_pacs_dir, args.k1_root, ["3of6"])
    build_root(base_pacs_dir, args.k3_root, ["1of6", "3of6", "5of6"])


if __name__ == "__main__":
    main()

"""
Build a volume-matched K=5 control (fix for the K-sweep volume confound): the
original K=1/K=3/K=5 comparison scales total image volume with K (50/150/250 per
pair), so any accuracy trend could be driven by volume rather than manifold
coverage - the same objection R1.4 raised about OT vs Diff-Src. This builds a K=5
variant subsampled to K=1's exact volume (50 images/pair, 150 images per held-out
target across its 3 relevant pairs) while still spanning all 5 interpolation
weights: for each of the 5 weight-folders, keep only 2 of the 10 conditioning
vectors (v00, v01; all 5 images each) instead of all 10 - 2 vectors x 5 images x 5
weights = 50 images/pair, matching K=1's 10 vectors x 5 images x 1 weight = 50.
If this still beats K=1, coverage (not volume) is driving the trend.

Usage:
  python prepare_k5_volmatched_dataroot.py --base_data_root .../data \
      --out_root .../data_k5_volmatched --n_vectors_per_weight 2
"""
import argparse
import os
import re

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
IMG_RE = re.compile(r"^img_(\d+)_v(\d+)\.jpg$")


def build_root(base_pacs_dir, out_root, n_vectors_per_weight):
    out_pacs = os.path.join(out_root, "pacs")
    os.makedirs(out_pacs, exist_ok=True)

    for d in DOMAINS:
        link_path = os.path.join(out_pacs, d)
        target = os.path.join(base_pacs_dir, d)
        if not os.path.exists(link_path):
            os.symlink(target, link_path)

    n_folders, n_images = 0, 0
    for folder in sorted(os.listdir(base_pacs_dir)):
        if not folder.startswith("SynDomain_OT_"):
            continue
        src_folder_dir = os.path.join(base_pacs_dir, folder)
        out_folder_dir = os.path.join(out_pacs, folder)
        os.makedirs(out_folder_dir, exist_ok=True)
        n_folders += 1

        for cls in sorted(os.listdir(src_folder_dir)):
            src_class_dir = os.path.join(src_folder_dir, cls)
            if not os.path.isdir(src_class_dir):
                continue
            out_class_dir = os.path.join(out_folder_dir, cls)
            os.makedirs(out_class_dir, exist_ok=True)

            for fname in sorted(os.listdir(src_class_dir)):
                m = IMG_RE.match(fname)
                if not m:
                    continue
                v_idx = int(m.group(2))
                if v_idx >= n_vectors_per_weight:
                    continue  # keep only the first n_vectors_per_weight conditioning vectors
                link_path = os.path.join(out_class_dir, fname)
                target = os.path.join(src_class_dir, fname)
                if not os.path.exists(link_path):
                    os.symlink(target, link_path)
                n_images += 1

    print(f"{out_root}: linked 4 original domains + {n_folders} OT synthetic domains "
          f"(subsampled to {n_vectors_per_weight} vectors/weight, {n_images} images total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--n_vectors_per_weight", type=int, default=2,
                     help="2 vectors x 5 images x 5 weights = 50 images/pair, matching K=1's volume")
    args = ap.parse_args()

    args.base_data_root = os.path.abspath(args.base_data_root)
    args.out_root = os.path.abspath(args.out_root)

    base_pacs_dir = os.path.join(args.base_data_root, "pacs")
    build_root(base_pacs_dir, args.out_root, args.n_vectors_per_weight)


if __name__ == "__main__":
    main()

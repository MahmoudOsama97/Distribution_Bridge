"""
Stage 1 - CLIP embedding extraction (revision plan Stage 1).

For all 4 PACS domains x 7 classes, extract unnormalized OpenCLIP ViT-H/14 image
embeddings (R^1024, paper convention Supp. S1) for every image, and cache per
domain-class to results/stage1_embeddings/{domain}_{class}.npz with:
  - "embeddings": (N, 1024) float32 matrix
  - "mean": (1024,) float32
  - "cov": (1024, 1024) float32
  - "n": int, sample count N_{i,c}
  - "paths": list of source image paths (str array), for traceability

Usage:
  python extract_embeddings.py --data_root /path/to/data/pacs \
      --out_dir /path/to/results/stage1_embeddings
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from distribution_bridge.generation.unclip_pipeline import load_image_encoder, encode_image  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_images(class_dir):
    if not os.path.isdir(class_dir):
        return []
    return sorted(
        os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(IMG_EXTS)
    )


def discover_domains(data_root):
    return sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))


def discover_classes(data_root, domains):
    """Union of class subdirs across all domains (DomainBed convention: consistent
    class set per dataset, but be robust to a domain missing a rare class)."""
    classes = set()
    for d in domains:
        domain_dir = os.path.join(data_root, d)
        classes.update(c for c in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, c)))
    return sorted(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="dir containing the dataset's domain folders")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--domains", default=None, help="comma-separated override; default: auto-discover")
    ap.add_argument("--classes", default=None, help="comma-separated override; default: auto-discover")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    domains = args.domains.split(",") if args.domains else discover_domains(args.data_root)
    classes = args.classes.split(",") if args.classes else discover_classes(args.data_root, domains)
    print(f"Domains ({len(domains)}): {domains}")
    print(f"Classes ({len(classes)}): {classes}")

    print(f"Loading OpenCLIP ViT-H/14 image encoder on {args.device} ({dtype})...")
    image_encoder, feature_extractor = load_image_encoder(device=args.device, dtype=dtype)

    total_images = 0
    for domain in domains:
        for cls in classes:
            out_path = os.path.join(args.out_dir, f"{domain}_{cls}.npz")
            if os.path.exists(out_path):
                print(f"SKIP (exists): {domain}/{cls}")
                continue

            class_dir = os.path.join(args.data_root, domain, cls)
            paths = list_images(class_dir)
            if not paths:
                print(f"WARNING: no images found for {domain}/{cls} at {class_dir}")
                continue

            embeds = []
            kept_paths = []
            n_skipped = 0
            for p in tqdm(paths, desc=f"{domain}/{cls}", leave=False):
                try:
                    img = Image.open(p)
                    emb = encode_image(img, image_encoder, feature_extractor, args.device, dtype)
                except Exception as e:
                    n_skipped += 1
                    print(f"WARNING: skipping unreadable image {p}: {type(e).__name__}: {e}")
                    continue
                embeds.append(emb.float().cpu().numpy()[0])
                kept_paths.append(p)
            if n_skipped:
                print(f"  {domain}/{cls}: skipped {n_skipped}/{len(paths)} unreadable image(s)")
            paths = kept_paths
            if not embeds:
                print(f"WARNING: no readable images for {domain}/{cls}, skipping")
                continue

            embeds = np.stack(embeds, axis=0)  # (N, 1024)
            mean = embeds.mean(axis=0)
            cov = np.cov(embeds, rowvar=False)

            np.savez(
                out_path,
                embeddings=embeds.astype(np.float32),
                mean=mean.astype(np.float32),
                cov=cov.astype(np.float32),
                n=embeds.shape[0],
                paths=np.array(paths),
            )
            total_images += embeds.shape[0]
            print(f"Saved {domain}/{cls}: N={embeds.shape[0]}, dim={embeds.shape[1]} -> {out_path}")

    print(f"\nStage 1 complete. Total images embedded this run: {total_images}")


if __name__ == "__main__":
    main()

"""
Stage 2a - OT synthetic PACS generation (revision plan Stage 2a).

For all 6 unordered PACS domain pairs x t in {1/6,...,5/6} x 7 classes:
  1. Sinkhorn-OT barycentric interpolation (ot_interpolate.py) between the pair's
     Stage-1 cached CLIP embeddings for that class.
  2. Sample 10 conditioning vectors without replacement from the interpolant set.
  3. Generate 5 images per vector (=50 images/class/weight) via the unCLIP pipeline.
  4. Save under data/pacs/SynDomain_OT_{domA}_{domB}_t{k}of6/{class}/img_*.jpg
     (folder naming is load-bearing for domainbed/datasets.py's synthetic-domain
     auto-detection/pruning: must contain "SynDomain" + both full domain names).
  5. Save the 10 conditioning vectors per (folder, class) to
     results/stage2_metadata/SynDomain_OT_.../  {class}_cond_embeds.npz
     for eps_gen (Stage 4 / E3) - kept OUT of data/pacs/ so it isn't picked up as a
     bogus extra "class" by torchvision.datasets.ImageFolder.

Usage:
  python generate_ot.py --embeddings_dir .../results/stage1_embeddings \
      --data_root .../data/pacs --metadata_dir .../results/stage2_metadata
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from distribution_bridge.generation.unclip_pipeline import load_pipeline  # noqa: E402
from distribution_bridge.generation.ot_interpolate import sinkhorn_barycentric_interpolants  # noqa: E402

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
WEIGHTS_K = [1, 2, 3, 4, 5]  # t = k/6

NEGATIVE_PROMPT = "ugly, blurry, malformed, deformed, noisy, text, watermark, signature"
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 10.0
NOISE_LEVEL = 10
SEED = 12345
N_COND_VECTORS = 10
N_IMAGES_PER_VECTOR = 5
SAVE_SIZE = 224
JPEG_QUALITY = 95


def load_class_embeddings(embeddings_dir, domain, cls):
    path = os.path.join(embeddings_dir, f"{domain}_{cls}.npz")
    data = np.load(path)
    return data["embeddings"]  # (N, 1024)


def save_image(pil_image, out_path):
    resized = pil_image.resize((SAVE_SIZE, SAVE_SIZE), Image.BILINEAR)
    resized.save(out_path, format="JPEG", quality=JPEG_QUALITY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--data_root", required=True, help="data/pacs - where SynDomain_* folders get written")
    ap.add_argument("--metadata_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pair_filter", default=None,
                     help="optional 'domA,domB' to restrict to one pair (for sharding across jobs)")
    ap.add_argument("--class_filter", default=None, help="optional single class name")
    args = ap.parse_args()

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"Loading unCLIP pipeline on {args.device} ({dtype})...")
    pipe, _, _ = load_pipeline(device=args.device, dtype=dtype)
    generator = torch.Generator(device=args.device).manual_seed(SEED)

    pairs = list(itertools.combinations(sorted(DOMAINS), 2))
    if args.pair_filter:
        a, b = args.pair_filter.split(",")
        pairs = [p for p in pairs if set(p) == {a, b}]
    classes = CLASSES if not args.class_filter else [args.class_filter]

    total_images = 0
    for domA, domB in pairs:
        for cls in classes:
            X = load_class_embeddings(args.embeddings_dir, domA, cls)
            Y = load_class_embeddings(args.embeddings_dir, domB, cls)

            for k in WEIGHTS_K:
                t = k / 6.0
                syn_folder = f"SynDomain_OT_{domA}_{domB}_t{k}of6"
                class_dir = os.path.join(args.data_root, syn_folder, cls)
                meta_dir = os.path.join(args.metadata_dir, syn_folder)
                meta_path = os.path.join(meta_dir, f"{cls}_cond_embeds.npz")

                if os.path.exists(meta_path):
                    existing = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")]) if os.path.isdir(class_dir) else 0
                    if existing >= N_COND_VECTORS * N_IMAGES_PER_VECTOR:
                        print(f"SKIP (complete): {syn_folder}/{cls}")
                        continue

                os.makedirs(class_dir, exist_ok=True)
                os.makedirs(meta_dir, exist_ok=True)

                z = sinkhorn_barycentric_interpolants(X, Y, t)  # (N_i, 1024)
                n_avail = z.shape[0]
                n_sample = min(N_COND_VECTORS, n_avail)
                rng = np.random.RandomState(SEED)
                sampled_idx = rng.choice(n_avail, size=n_sample, replace=False)
                cond_vectors = z[sampled_idx]  # (n_sample, 1024)

                np.savez(meta_path, embeds=cond_vectors.astype(np.float32),
                         source_domain=domA, target_domain=domB, weight_t=t, weight_k=k, cls=cls)

                clean_cls = cls.replace("_", " ")
                prompt = f"a high-quality photo of a {clean_cls}"

                for v_idx, cond_vec in enumerate(cond_vectors):
                    out_files = [
                        os.path.join(class_dir, f"img_{img_idx:02d}_v{v_idx:02d}.jpg")
                        for img_idx in range(N_IMAGES_PER_VECTOR)
                    ]
                    if all(os.path.exists(f) for f in out_files):
                        continue

                    cond_tensor = torch.from_numpy(cond_vec).to(args.device, dtype=dtype).unsqueeze(0)
                    images = pipe(
                        image_embeds=cond_tensor,
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        noise_level=NOISE_LEVEL,
                        num_images_per_prompt=N_IMAGES_PER_VECTOR,
                        generator=generator,
                        output_type="pil",
                    ).images

                    for img, out_path in zip(images, out_files):
                        save_image(img, out_path)
                        total_images += 1

                print(f"Done: {syn_folder}/{cls} ({n_sample} vectors x {N_IMAGES_PER_VECTOR} imgs)")

    print(f"\nStage 2a complete. Total images generated this run: {total_images}")


if __name__ == "__main__":
    main()

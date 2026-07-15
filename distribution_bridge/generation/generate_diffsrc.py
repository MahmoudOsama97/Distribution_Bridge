"""
Stage 2b - Diff-Src no-interpolation control generation (revision plan Stage 2b).
Answers R1.4 / the AE's core disentanglement point: identical pipeline, generator,
prompts, and image volume as Stage 2a - the ONLY difference is the conditioning
distribution has no OT / no interpolation (conditioning at t=0 and t=1 only, i.e.
directly on raw source-domain embeddings).

For each of the 6 unordered PACS domain pairs x 7 classes:
  - 5 "pseudo-weight slots" (parity with OT's 5 t-values), each slot samples 5 fresh
    conditioning vectors (without replacement within the slot) from each endpoint
    domain's raw Stage-1 embeddings, generates 5 images/vector.
  - Total: 5 slots x (5 vecs from domA + 5 vecs from domB) x 5 imgs/vec
         = 250 images/class/pair (125 domA-conditioned + 125 domB-conditioned),
    exactly matching Stage 2a's 5 weights x 10 vectors x 5 imgs = 250/class/pair.
  - Saved to data/pacs/SynDomain_DiffSrc_{domA}_{domB}/{class}/img_*.jpg (single
    folder per pair, no per-weight subfolders, since there's no interpolation axis).

Usage mirrors generate_ot.py.
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

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
N_SLOTS = 5          # parity with OT's 5 t-weights
N_VECTORS_PER_ENDPOINT_PER_SLOT = 5
N_IMAGES_PER_VECTOR = 5

NEGATIVE_PROMPT = "ugly, blurry, malformed, deformed, noisy, text, watermark, signature"
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 10.0
NOISE_LEVEL = 10
SEED = 12345
SAVE_SIZE = 224
JPEG_QUALITY = 95


def load_class_embeddings(embeddings_dir, domain, cls):
    data = np.load(os.path.join(embeddings_dir, f"{domain}_{cls}.npz"))
    return data["embeddings"]


def save_image(pil_image, out_path):
    pil_image.resize((SAVE_SIZE, SAVE_SIZE), Image.BILINEAR).save(out_path, format="JPEG", quality=JPEG_QUALITY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--metadata_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pair_filter", default=None)
    ap.add_argument("--class_filter", default=None)
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
            Xa = load_class_embeddings(args.embeddings_dir, domA, cls)
            Xb = load_class_embeddings(args.embeddings_dir, domB, cls)

            syn_folder = f"SynDomain_DiffSrc_{domA}_{domB}"
            class_dir = os.path.join(args.data_root, syn_folder, cls)
            meta_dir = os.path.join(args.metadata_dir, syn_folder)
            os.makedirs(class_dir, exist_ok=True)
            os.makedirs(meta_dir, exist_ok=True)

            clean_cls = cls.replace("_", " ")
            prompt = f"a high-quality photo of a {clean_cls}"

            all_meta_vectors = []
            all_meta_labels = []

            for slot in range(N_SLOTS):
                rng = np.random.RandomState(SEED + slot)
                for endpoint_name, X in (("A", Xa), ("B", Xb)):
                    n_avail = X.shape[0]
                    n_sample = min(N_VECTORS_PER_ENDPOINT_PER_SLOT, n_avail)
                    idx = rng.choice(n_avail, size=n_sample, replace=False)
                    cond_vectors = X[idx]

                    for v_idx, cond_vec in enumerate(cond_vectors):
                        all_meta_vectors.append(cond_vec)
                        all_meta_labels.append(f"slot{slot}_end{endpoint_name}_v{v_idx:02d}")

                        out_files = [
                            os.path.join(class_dir, f"img_{img_idx:02d}_slot{slot}_end{endpoint_name}_v{v_idx:02d}.jpg")
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

            meta_path = os.path.join(meta_dir, f"{cls}_cond_embeds.npz")
            np.savez(
                meta_path,
                embeds=np.stack(all_meta_vectors).astype(np.float32),
                labels=np.array(all_meta_labels),
                source_domain=domA, target_domain=domB, cls=cls,
            )
            print(f"Done: {syn_folder}/{cls} ({len(all_meta_vectors)} vectors x {N_IMAGES_PER_VECTOR} imgs)")

    print(f"\nStage 2b complete. Total images generated this run: {total_images}")


if __name__ == "__main__":
    main()

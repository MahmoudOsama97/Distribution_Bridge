"""
Stage 0 smoke test, parts (a)+(b) of the revision plan's gate:
  (a) unCLIP pipeline + CLIP image encoder load in FP16 on 1 GPU.
  (b) encode 10 PACS images -> 1024-d embeddings; generate 2 images from one
      embedding; re-encode; print cosine similarity (expect > 0.7).
Part (c) (100-step ERM smoke run) is a separate `domainbed.scripts.train` call,
launched from the same sbatch job right after this script exits 0.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from distribution_bridge.generation.unclip_pipeline import load_pipeline, encode_image  # noqa: E402

DATA_ROOT = os.environ.get("DDB_DATA_ROOT", "./data/pacs")
DOMAIN = "photo"
CLASS = "dog"
N_PROBE_IMAGES = 10
SEED = 12345


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Smoke test must run on a GPU node."
    dtype = torch.float16

    print(f"[a] Loading unCLIP pipeline + CLIP image encoder on {device} ({dtype})...")
    pipe, image_encoder, feature_extractor = load_pipeline(device=device, dtype=dtype)
    print("[a] PASS: pipeline loaded.")

    class_dir = os.path.join(DATA_ROOT, DOMAIN, CLASS)
    paths = sorted(
        os.path.join(class_dir, f) for f in os.listdir(class_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )[:N_PROBE_IMAGES]
    assert len(paths) >= 1, f"No images found in {class_dir}"
    print(f"[b] Encoding {len(paths)} images from {DOMAIN}/{CLASS}...")

    embeds = []
    for p in paths:
        img = Image.open(p)
        emb = encode_image(img, image_encoder, feature_extractor, device, dtype)
        embeds.append(emb.float().cpu().numpy()[0])
    embeds = np.stack(embeds, axis=0)
    print(f"[b] Embedding shape: {embeds.shape} (expect (*, 1024))")
    assert embeds.shape[1] == 1024, f"Unexpected embedding dim {embeds.shape[1]}, expected 1024"

    cond_vec = embeds[0]
    cond_tensor = torch.from_numpy(cond_vec).to(device, dtype=dtype).unsqueeze(0)

    generator = torch.Generator(device=device).manual_seed(SEED)
    print("[b] Generating 2 images from one embedding (40 steps, guidance 10, noise_level 10)...")
    images = pipe(
        image_embeds=cond_tensor,
        prompt="a high-quality photo of a dog",
        negative_prompt="ugly, blurry, malformed, deformed, noisy, text, watermark, signature",
        num_inference_steps=40,
        guidance_scale=10.0,
        noise_level=10,
        num_images_per_prompt=2,
        generator=generator,
        output_type="pil",
    ).images
    assert len(images) == 2, f"Expected 2 images, got {len(images)}"

    out_dir = os.environ.get("DDB_OUT_DIR", "./results/sample_images/smoke_test")
    os.makedirs(out_dir, exist_ok=True)
    sims = []
    for i, img in enumerate(images):
        img_path = os.path.join(out_dir, f"gen_{i}.jpg")
        img.resize((224, 224), Image.BILINEAR).save(img_path, quality=95)
        re_emb = encode_image(img, image_encoder, feature_extractor, device, dtype).float().cpu().numpy()[0]
        sim = cosine_sim(cond_vec, re_emb)
        sims.append(sim)
        print(f"[b] Generated image {i}: re-encoded cosine similarity to conditioning embedding = {sim:.4f}")

    mean_sim = float(np.mean(sims))
    print(f"[b] Mean cosine similarity: {mean_sim:.4f} (gate: > 0.7)")
    if mean_sim > 0.7:
        print("[b] PASS")
    else:
        print("[b] FAIL: cosine similarity below 0.7 threshold")
        sys.exit(1)

    print("\n=== Stage 0 parts (a) + (b): ALL PASS ===")


if __name__ == "__main__":
    main()

"""
One-off diagnostic: compare the scale of embeddings from (a) the diffusers
SD-2.1-unCLIP image_encoder (what Stage 1 currently uses) vs (b) native open_clip
ViT-H-14/laion2b_s32b_b79k .encode_image(), to explain why measured D_max (~24.5)
is ~24x the paper's reported PACS D_max (~1.0-1.1, Table 7).
"""
import os
import sys

import numpy as np
import open_clip
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from distribution_bridge.generation.unclip_pipeline import load_image_encoder, encode_image  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def main():
    photo_dir = "data/pacs/photo/dog"
    sketch_dir = "data/pacs/sketch/dog"
    photo_paths = [os.path.join(photo_dir, f) for f in sorted(os.listdir(photo_dir))[:15]]
    sketch_paths = [os.path.join(sketch_dir, f) for f in sorted(os.listdir(sketch_dir))[:15]]

    print("=== (a) diffusers unCLIP image_encoder (current Stage 1 method) ===")
    image_encoder, feature_extractor = load_image_encoder(device=DEVICE, dtype=DTYPE)
    photo_a = np.stack([
        encode_image(Image.open(p), image_encoder, feature_extractor, DEVICE, DTYPE).float().cpu().numpy()[0]
        for p in photo_paths
    ])
    sketch_a = np.stack([
        encode_image(Image.open(p), image_encoder, feature_extractor, DEVICE, DTYPE).float().cpu().numpy()[0]
        for p in sketch_paths
    ])
    print("photo norm mean:", np.linalg.norm(photo_a, axis=1).mean())
    print("sketch norm mean:", np.linalg.norm(sketch_a, axis=1).mean())
    print("mean-embedding distance:", np.linalg.norm(photo_a.mean(0) - sketch_a.mean(0)))
    del image_encoder

    print("\n=== (b) native open_clip ViT-H-14 / laion2b_s32b_b79k ===")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=os.environ.get("DDB_OPEN_CLIP_CACHE", os.path.expanduser("~/.cache/open_clip"))
    )
    model = model.to(DEVICE).eval()

    def embed_native(path):
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model.encode_image(img)
        return feat[0].float().cpu().numpy()

    photo_b = np.stack([embed_native(p) for p in photo_paths])
    sketch_b = np.stack([embed_native(p) for p in sketch_paths])
    print("photo norm mean:", np.linalg.norm(photo_b, axis=1).mean())
    print("sketch norm mean:", np.linalg.norm(sketch_b, axis=1).mean())
    print("mean-embedding distance:", np.linalg.norm(photo_b.mean(0) - sketch_b.mean(0)))

    print("\n=== (c) native open_clip, L2-NORMALIZED (typical CLIP-similarity convention) ===")
    photo_b_norm = photo_b / np.linalg.norm(photo_b, axis=1, keepdims=True)
    sketch_b_norm = sketch_b / np.linalg.norm(sketch_b, axis=1, keepdims=True)
    print("mean-embedding distance (normalized):", np.linalg.norm(photo_b_norm.mean(0) - sketch_b_norm.mean(0)))

    print("\n=== (d) cross-check: cosine sim between (a) and (b) for same image (are they the same underlying features, just rescaled?) ===")
    a0 = photo_a[0] / np.linalg.norm(photo_a[0])
    b0 = photo_b[0] / np.linalg.norm(photo_b[0])
    print("cosine sim (a vs b, first photo/dog image):", float(np.dot(a0, b0)))


if __name__ == "__main__":
    main()

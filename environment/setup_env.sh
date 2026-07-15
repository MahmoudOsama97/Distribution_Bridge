#!/bin/bash
# Build the Distribution Bridge Python environment. Run once on a machine with
# internet access (a cluster login node, typically) - the HF cache populated
# here can then be reused offline (HF_HUB_OFFLINE=1) from compute jobs, since
# many SLURM clusters have no internet access on compute nodes.
#
# Configure via environment variables before running, e.g.:
#   export DDB_VENV=/path/to/ddb_env
#   export DDB_HF_HOME=/path/to/hf_cache
#   export DDB_TORCH_HOME=/path/to/torch_cache
#   ./environment/setup_env.sh
set -euo pipefail

ENV_DIR="${DDB_VENV:-$HOME/ddb_env}"
HF_HOME="${DDB_HF_HOME:-$HOME/.cache/huggingface}"
TORCH_HOME="${DDB_TORCH_HOME:-$HOME/.cache/torch}"

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip

# torch 2.1.2 PyPI linux wheels bundle their own CUDA 12.1 runtime deps, so this
# works against the driver alone (no separate CUDA toolkit install required).
pip install torch==2.1.2 torchvision==0.16.2

pip install \
  diffusers==0.21.0 \
  transformers==4.33.0 \
  accelerate==0.23.0 \
  huggingface_hub==0.18.0 \
  POT==0.9.0 \
  open_clip_torch==2.23.0 \
  Pillow tqdm numpy scipy pandas matplotlib seaborn scikit-learn \
  safetensors sentencepiece ftfy \
  wilds==2.0.0 einops geomloss

# POT==0.9.0's optim.py imports scipy.optimize.scalar_search_armijo, removed in
# scipy>=1.12. Pin the newest scipy that still has it.
pip install --force-reinstall --no-deps scipy==1.11.2

# backpack-for-pytorch==1.3.0 (used only by algorithms.Fishr) hard-pins torch<2.0
# and is incompatible with torch>=2.1's changed nn.grad API. domainbed/algorithms.py
# already guards its import in a try/except (falls back to backpack=None), so it is
# intentionally omitted here rather than pinned against an unsupported torch version.
# Install separately (accepting the Fishr-only limitation) if needed:
#   pip install --no-deps backpack-for-pytorch==1.3.0

echo "=== Versions ==="
python - <<'PY'
import torch, diffusers, transformers, ot, open_clip
print("torch", torch.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("POT", ot.__version__)
PY

# --- Pre-download model weights into an offline-reusable cache ---
# stabilityai/stable-diffusion-2-1-unclip was removed/renamed on the Hub; the
# ungated mirror diffusers/stable-diffusion-2-1-unclip-i2i-h has an identical file
# layout (image_encoder/image_normalizer/unet/vae/etc.). If your account needs a
# token for any gated resources, run `huggingface-cli login` first.
mkdir -p "$HF_HOME"
HF_HOME="$HF_HOME" python - <<'PY'
from huggingface_hub import snapshot_download
print("Downloading diffusers/stable-diffusion-2-1-unclip-i2i-h ...")
snapshot_download("diffusers/stable-diffusion-2-1-unclip-i2i-h")
print("Done.")
PY

HF_HOME="$HF_HOME" python - <<'PY'
import open_clip
print("Downloading OpenCLIP ViT-H-14 / laion2b_s32b_b79k ...")
open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
print("Done.")
PY

# domainbed/networks.py loads torchvision.models.resnet50(pretrained=True), which
# hits download.pytorch.org via torch.hub - pre-cache it too if compute nodes have
# no internet access.
mkdir -p "$TORCH_HOME"
TORCH_HOME="$TORCH_HOME" python - <<'PY'
import torchvision.models
print("Downloading torchvision resnet50 ImageNet weights...")
torchvision.models.resnet50(pretrained=True)
print("Done.")
PY

echo "=== setup_env.sh complete ==="
echo "Venv: $ENV_DIR"
echo "HF_HOME: $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"

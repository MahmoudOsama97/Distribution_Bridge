#!/bin/bash
# Example SLURM job: Stage 1 - CLIP embedding extraction.
# Edit the #SBATCH lines and the DDB_* environment variables for your cluster/account,
# or export them before calling `sbatch` so this script picks them up unchanged.
#SBATCH --account=${DDB_ACCOUNT:?set DDB_ACCOUNT, e.g. export DDB_ACCOUNT=your_slurm_account}
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=ddb_extract_embeddings
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -euo pipefail

module load python/3.11 cuda/12.2 cudnn
source "${DDB_VENV:-$HOME/ddb_env}/bin/activate"

export HF_HOME="${DDB_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1

REPO_ROOT="${DDB_REPO_ROOT:?set DDB_REPO_ROOT to the path containing distribution_bridge/}"
DATA_ROOT="${DDB_DATA_ROOT:-$REPO_ROOT/data/pacs}"
OUT_DIR="${DDB_OUT_DIR:-$REPO_ROOT/results/stage1_embeddings}"

cd "$REPO_ROOT"

python -m distribution_bridge.generation.extract_embeddings \
  --data_root "$DATA_ROOT" \
  --out_dir "$OUT_DIR"

echo "=== Stage 1 embedding extraction complete ==="

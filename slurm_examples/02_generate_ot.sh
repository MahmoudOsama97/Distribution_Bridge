#!/bin/bash
# Example SLURM job: Stage 2a - Sinkhorn-OT interpolation + unCLIP generation.
# Produces SynDomain_OT_<domA>_<domB>_t<k>of6/<class>/*.jpg synthetic domains.
# Edit the #SBATCH lines and DDB_* environment variables for your cluster/account.
#SBATCH --account=${DDB_ACCOUNT:?set DDB_ACCOUNT}
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=ddb_generate_ot
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -euo pipefail

module load python/3.11 cuda/12.2 cudnn
source "${DDB_VENV:-$HOME/ddb_env}/bin/activate"

export HF_HOME="${DDB_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1

REPO_ROOT="${DDB_REPO_ROOT:?set DDB_REPO_ROOT}"
DATA_ROOT="${DDB_DATA_ROOT:-$REPO_ROOT/data/pacs}"

cd "$REPO_ROOT"

# Generates all 6 unordered domain pairs x 5 interpolation weights x all classes by
# default; pass --pair_filter "domA_domB" / --class_filter <cls> to run a subset
# (useful for splitting the ~6h full run across several jobs).
python -m distribution_bridge.generation.generate_ot \
  --embeddings_dir "$REPO_ROOT/results/stage1_embeddings" \
  --data_root "$DATA_ROOT" \
  --metadata_dir "$REPO_ROOT/results/stage2_metadata"

echo "=== Stage 2a (OT) generation complete ==="

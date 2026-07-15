#!/bin/bash
# Example SLURM job: Stage 3 - DomainBed training with DDB's two-tiered domain
# batching (k_syn_domains_per_step synthetic-domain mini-batches drawn alongside
# every original-domain mini-batch each step; see domainbed/scripts/train.py and
# the paper's Sec. 3.5.4). data_dir must contain the original domain folders plus
# SynDomain_* folders produced by Stage 2a/2b - datasets.py auto-discovers them and
# auto-prunes any synthetic pair touching the held-out test domain.
# Edit the #SBATCH lines, DDB_* variables, and the algorithm/test_envs/seed below.
#SBATCH --account=${DDB_ACCOUNT:?set DDB_ACCOUNT}
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=ddb_train
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -euo pipefail

module load python/3.11 cuda/12.2 cudnn
source "${DDB_VENV:-$HOME/ddb_env}/bin/activate"

export HF_HOME="${DDB_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TORCH_HOME="${DDB_TORCH_HOME:-$HOME/.cache/torch}"

REPO_ROOT="${DDB_REPO_ROOT:?set DDB_REPO_ROOT}"
cd "$REPO_ROOT"

ALGORITHM="${DDB_ALGORITHM:-ERM}"
TEST_ENV="${DDB_TEST_ENV:-0}"        # index into the dataset's domain ordering
SEED="${DDB_SEED:-1}"
DATA_DIR="${DDB_TRAIN_DATA_DIR:-$REPO_ROOT/data}"  # original domains + SynDomain_* folders
OUTPUT_DIR="${DDB_TRAIN_OUTPUT_DIR:-$REPO_ROOT/results/raw/${ALGORITHM}_env${TEST_ENV}_seed${SEED}}"

python -m domainbed.scripts.train \
  --data_dir "$DATA_DIR" \
  --dataset PACS \
  --algorithm "$ALGORITHM" \
  --test_envs "$TEST_ENV" \
  --seed "$SEED" \
  --trial_seed "$SEED" \
  --hparams '{"k_syn_domains_per_step":3,"lr":5e-5,"batch_size":32,"weight_decay":0,"resnet50_augmix":false}' \
  --output_dir "$OUTPUT_DIR"

echo "=== Training complete: $ALGORITHM, test_env=$TEST_ENV, seed=$SEED ==="

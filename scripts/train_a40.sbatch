#!/bin/bash
# Train the ILP-PFN on one A40. Adjust partition/account/module lines to your cluster.
#SBATCH --job-name=ilp-pfn
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
##SBATCH --partition=<gpu-partition>
##SBATCH --account=<account>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
# module load python/3.11 cuda/12.x       # if your cluster uses modules
source .venv/bin/activate

RUN=${RUN:-runs/base}
# Resumes automatically from $RUN/ckpt.pt, so the same script can be resubmitted after a timeout.
python -m ilp_pfn.train \
  --out_dir "$RUN" \
  --d_model 384 --n_layers 8 --n_heads 6 \
  --steps 100000 --batch_size 256 --lr 6e-4 --warmup 1000 \
  --num_workers $((SLURM_CPUS_PER_TASK - 2)) \
  --compile 1 "$@"

python -m ilp_pfn.evaluate --ckpt "$RUN/ckpt.pt" --n 500 --k 16

#!/bin/bash
# Full RetiLink training on one local GPU (sequential seeds), same outputs as scripts/JobSubmit.sh.
#   GPU=0 RETILINK_EXP_NO=1 bash scripts/train_local.sh
set -e
cd "$(dirname "$0")/.."
PY=/home/users/nshaik3/miniconda3/envs/remote-dip-env/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SLURM_CPUS_PER_TASK=8
export RETILINK_EXP_NO=${RETILINK_EXP_NO:-1}
for s in 42 43; do
  [ -f /data/users3/nshaik3/Projects/Oculomics/RetiLink/$RETILINK_EXP_NO/features/seed${s}_test_logits.npy ] || $PY -m retilink.ml.train_image --seed $s
done
$PY -m retilink.ml.evaluate
$PY -m retilink.ml.train_systemic

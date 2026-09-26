#!/bin/bash
#SBATCH -p qTRDGPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:A40:2
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -e /data/users3/nshaik3/Projects/Oculomics/RetiLink/logs/error%A.err
#SBATCH -o /data/users3/nshaik3/Projects/Oculomics/RetiLink/logs/out%A.out
#SBATCH -A trends517s113
#SBATCH --oversubscribe
#SBATCH -J Insight Rx
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=nshaik3@student.gsu.edu
#
# Insight Rx training on mBRSET (from the login node):
#   ssh nshaik3@arctrdlogin001.rs.gsu.edu
#   cd /home/users/nshaik3/Desktop/Oculomics/RetiLink && sbatch scripts/JobSubmit.sh
#   INSIGHTRX_EXP_NO=2 sbatch --export=ALL scripts/JobSubmit.sh      # fresh output dir
#   INSIGHTRX_SMOKE=1 sbatch --export=ALL scripts/JobSubmit.sh       # quick end-to-end check
# Outputs: /data/users3/nshaik3/Projects/Oculomics/RetiLink/<INSIGHTRX_EXP_NO>/

sleep 5s
source /home/users/nshaik3/miniconda3/bin/activate remote-dip-env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-12}
export OMP_NUM_THREADS=2
export INSIGHTRX_EXP_NO=${INSIGHTRX_EXP_NO:-1}
# two data-loader pools share the node's CPUs
export SLURM_CPUS_PER_TASK=$(( SLURM_CPUS_PER_TASK / 2 ))

cd /home/users/nshaik3/Desktop/Oculomics/RetiLink
echo "Insight Rx job $SLURM_JOB_ID on $(hostname), exp $INSIGHTRX_EXP_NO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# one seed per GPU, in parallel -> 2-model ensemble
CUDA_VISIBLE_DEVICES=0 python3 -m insightrx.ml.train_image --seed 42 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 -m insightrx.ml.train_image --seed 43 &
P1=$!
wait $P0; S0=$?
wait $P1; S1=$?
[ $S0 -ne 0 ] && [ $S1 -ne 0 ] && { echo "both seeds failed"; exit 1; }

python3 -m insightrx.ml.evaluate && python3 -m insightrx.ml.train_systemic
status=$?
sleep 5s
exit $status

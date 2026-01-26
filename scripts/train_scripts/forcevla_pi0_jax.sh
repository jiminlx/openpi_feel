#!/bin/bash
#SBATCH --job-name=pi0_forcevla_jax_gripper_tactile_4tasks
#SBATCH --partition=sjw_alinlab_h100
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --output=slurm_out/%j_%x.out
#SBATCH --error=slurm_out/%j_%x.err

CKPT_NAME="pi0_forcevla_jax_gripper_tactile_4tasks"
CKPT_PATH="/sjw_alinlab2/home/myungkyu/workspace/TactileVLA/TactilePi05/ckpt/tactile_vla/$CKPT_NAME"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py forcevla_pi0_gripper_tactile \
    --exp-name=$CKPT_NAME \
    --batch-size=16 \
    --num-workers=12 \
    --save-interval=10000 \
    --num-train-steps=30000 \
    --checkpoint-base-dir=$CKPT_PATH \
    --overwrite
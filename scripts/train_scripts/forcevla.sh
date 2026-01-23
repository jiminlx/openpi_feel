#!/bin/bash
#SBATCH --job-name=pi05_forcevla_gripper_tactile_4tasks
#SBATCH --partition=sjw_alinlab_h100
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --output=slurm_out/%j_%x.out
#SBATCH --error=slurm_out/%j_%x.err

CKPT_NAME="pi05_forcevla_gripper_tactile_4tasks"
CKPT_PATH="/sjw_alinlab2/home/myungkyu/workspace/TactileVLA/TactilePi05/ckpt/tactile_vla/$CKPT_NAME"

uv run scripts/train_pytorch.py forcevla_gripper_tactile \
    --exp_name $CKPT_NAME \
    --batch-size 16 \
    --num_workers 12 \
    --save_interval 10000 \
    --num_train_steps 30000 \
    --checkpoint-base-dir $CKPT_PATH
#!/bin/bash
#SBATCH --job-name=pi0_jax_unified
#SBATCH --partition=sjw_alinlab_h100
#SBATCH --nodelist=worker-node1002
#SBATCH --nodes=1                     
#SBATCH --gpus=2
#SBATCH --output=slurm_train_logs/%x_%j.out
#SBATCH --error=slurm_train_logs/%x_%j.err

CONFIG_NAME="pi0_decoupled_stream_gripper_tactile"
EXP_NAME="pi0_decoupled_stream_gripper_tactile_2stage"
CHECKPOINT_DIR="/sjw_alinlab2/home/jimin/openpi_feel/checkpoints_icml_jax"
NUM_WORKERS=8
BATCH_SIZE=8

mkdir -p slurm_train_logs

echo "=================================================================="
echo "Starting Unified 2-Stage Training"
echo "=================================================================="

# We do NOT use --freeze_action_stream here because the python script handles it internally
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/unified_train_2stage.py $CONFIG_NAME \
    --exp_name $EXP_NAME \
    --batch-size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --save_interval 5000 \
    --num_train_steps 30000 \
    --checkpoint_base_dir $CHECKPOINT_DIR \
    --overwrite \
    --fsdp-devices 2

echo "Job finished at $(date)"
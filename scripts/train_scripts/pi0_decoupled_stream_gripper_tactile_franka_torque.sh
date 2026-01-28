#!/bin/bash
#SBATCH --job-name=pi0_jax_decoupled_tactile
#SBATCH --partition=sjw_alinlab_h100
#SBATCH --nodelist=worker-node1002
#SBATCH --nodes=1                     
#SBATCH --gpus=2
#SBATCH --output=slurm_train_logs/%x_%j.out
#SBATCH --error=slurm_train_logs/%x_%j.err

# ---------------------------------------------------------------------------
# [Common Settings]
# ---------------------------------------------------------------------------
CONFIG_NAME="pi0_decoupled_stream_gripper_tactile_franka_torque"
CHECKPOINT_DIR="/sjw_alinlab2/home/jimin/openpi_feel/checkpoints_icml_jax"
NUM_WORKERS=12
BATCH_SIZE=8

# Create log dir
mkdir -p slurm_train_logs


# ---------------------------------------------------------------------------
# [Stage 1] Tactile&Torque Stream Pre-training (Action Freeze)
# ---------------------------------------------------------------------------
STAGE1_EXP_NAME="pi0_decoupled_stream_gripper_tactile_franka_torque_stage1"
STAGE1_STEPS=5000
STAGE1_TACTILE_WEIGHT=0.3   
STAGE1_TORQUE_WEIGHT=0.3

echo "=================================================================="
echo "Starting Stage 1: Tactile&Torque Pre-training (0 ~ $STAGE1_STEPS steps)"
echo "Action Stream will be FROZEN."
echo "=================================================================="

# Note: Added --freeze_action_stream flag which is handled in train.py
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_NAME \
    --exp_name $STAGE1_EXP_NAME \
    --batch-size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --save_interval 5000 \
    --num_train_steps $STAGE1_STEPS \
    --checkpoint-base-dir $CHECKPOINT_DIR \
    --model.loss_tactile_weight $STAGE1_TACTILE_WEIGHT \
    --model.loss_torque_weight $STAGE1_TORQUE_WEIGHT \
    --freeze_action_stream \
    --overwrite \
    --fsdp-devices 2 

# Check Success
if [ $? -ne 0 ]; then
    echo "Stage 1 failed! Exiting..."
    exit 1
fi

# ---------------------------------------------------------------------------
# [Stage 2] Joint Fine-tuning (Unfreeze All)
# ---------------------------------------------------------------------------
STAGE2_EXP_NAME="pi0_decoupled_stream_gripper_tactile_franka_torque_stage2"
TOTAL_STEPS=30000  # 5k(Stage1) + 20k(Add) = 25k Total
STAGE2_TACTILE_WEIGHT=0.05 
STAGE2_TORQUE_WEIGHT=0.05

# Path to the last checkpoint of Stage 1 (Step 4999 or 5000 depending on save logic)
# Note: Checkpoints in Orbax are usually named by step number.
# If save_interval is 5000, and we ran 5000 steps (0-4999), the save might be at 4999 or 5000.
# Assuming standard logic saves at num_train_steps - 1 or num_train_steps.
# Let's target the exact folder.
INT_STAGE1_CKPT_STEP=$((STAGE1_STEPS - 1))
STAGE1_CKPT_PATH="$CHECKPOINT_DIR/$CONFIG_NAME/$STAGE1_EXP_NAME/$INT_STAGE1_CKPT_STEP"

echo "=================================================================="
echo "Starting Stage 2: Joint Training (0 ~ $TOTAL_STEPS steps)"
echo "Loading weights from: $STAGE1_CKPT_PATH"
echo "Action Stream will be UNFROZEN (Optimizer Reset)."
echo "=================================================================="

# Note: We do NOT use --resume. instead we use --checkpoint_path to just load params.
# We do NOT pass --freeze_action_stream here.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 uv run scripts/train.py $CONFIG_NAME \
    --exp_name $STAGE2_EXP_NAME \
    --batch-size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --save_interval 5000 \
    --num_train_steps $TOTAL_STEPS \
    --checkpoint_base_dir $CHECKPOINT_DIR \
    --model.loss_tactile_weight $STAGE2_TACTILE_WEIGHT \
    --model.loss_torque_weight $STAGE2_TORQUE_WEIGHT \
    --checkpoint_path $STAGE1_CKPT_PATH \
    --overwrite \
    --fsdp-devices 2 

echo "Job finished at $(date)"
#!/bin/bash
#SBATCH --job-name=pi05_gripper_tactile_franka_torque_2stage
#SBATCH --partition=sjw_alinlab_h100
#SBATCH --nodelist=worker-node1002
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --output=slurm_train_logs/%x_%j.out
#SBATCH --error=slurm_train_logs/%x_%j.err

# ---------------------------------------------------------------------------
# [공통 설정]
# ---------------------------------------------------------------------------
CONFIG_NAME="decoupled_stream_gripper_tactile_franka_torque"
CHECKPOINT_DIR="/sjw_alinlab2/home/jimin/openpi_feel/checkpoints_icml"
NUM_WORKERS=8
BATCH_SIZE=16

# 로그 폴더 생성
mkdir -p slurm_train_logs

# 포트 충돌 방지를 위한 랜덤 포트 설정 (DDP 사용 시 중요)
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)
echo "Using Master Port: $MASTER_PORT"

# ---------------------------------------------------------------------------
# [Stage 1] Gripper-Tactile-Franka-Torque Stream Pre-training (Action Freeze)
# ---------------------------------------------------------------------------
STAGE1_EXP_NAME="decoupled_stream_gripper_tactile_franka_torque_stage1"
STAGE1_STEPS=5000
# Stage 1에서는 Tactile Loss가 지배적이도록 설정 (어차피 Action은 Freeze라 0임)
STAGE1_TORQUE_WEIGHT=1.0
STAGE1_TACTILE_WEIGHT=1.0

echo "=================================================================="
echo "Starting Stage 1: Gripper-Tactile-Franka-Torque Pre-training (0 ~ $STAGE1_STEPS steps)"
echo "Action Stream will be FROZEN."
echo "=================================================================="

uv run scripts/train_pytorch.py $CONFIG_NAME \
    --exp_name $STAGE1_EXP_NAME \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --save_interval 5000 \
    --num_train_steps $STAGE1_STEPS \
    --checkpoint_base_dir $CHECKPOINT_DIR \
    --model.loss_torque_weight $STAGE1_TORQUE_WEIGHT \
    --model.loss_tactile_weight $STAGE1_TACTILE_WEIGHT \
    --freeze_action_stream


# Stage 1 성공 여부 확인
if [ $? -ne 0 ]; then
    echo "Stage 1 failed! Exiting..."
    exit 1
fi

# ---------------------------------------------------------------------------
# [Stage 2] Gripper-Tactile-Franka-Torque Joint Fine-tuning (Unfreeze All)
# ---------------------------------------------------------------------------
STAGE2_EXP_NAME="decoupled_stream_gripper_tactile_franka_torque_stage2"
TOTAL_STEPS=25000  # 10k(Stage1) + 20k(Add) = 30k Total
STAGE2_TORQUE_WEIGHT=0.05 # Joint 학습 시 가중치 조절
STAGE2_TACTILE_WEIGHT=0.05 # Joint 학습 시 가중치 조절

# Stage 1에서 저장된 마지막 체크포인트 경로 (Stage1 Steps -1 step)
INT_STAGE1_CKPT_PATH=$((STAGE1_STEPS-1))
STAGE1_CKPT_PATH="$CHECKPOINT_DIR/$CONFIG_NAME/$STAGE1_EXP_NAME/$INT_STAGE1_CKPT_PATH"


echo "=================================================================="
echo "Starting Stage 2: Gripper-Tactile-Franka-Torque Joint Training (0 ~ $TOTAL_STEPS steps)"
echo "Loading weights from: $STAGE1_CKPT_PATH"
echo "Action Stream will be UNFROZEN (Optimizer Reset)."
echo "=================================================================="

# 주의: resume 플래그는 끄고, pytorch_weight_path를 사용해야 Optimizer가 리셋됨
uv run scripts/train_pytorch.py $CONFIG_NAME \
    --exp_name $STAGE2_EXP_NAME \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --save_interval 5000 \
    --num_train_steps $TOTAL_STEPS \
    --checkpoint_base_dir $CHECKPOINT_DIR \
    --model.loss_torque_weight $STAGE2_TORQUE_WEIGHT \
    --model.loss_tactile_weight $STAGE2_TACTILE_WEIGHT \
    --pytorch_weight_path $STAGE1_CKPT_PATH

echo "All Stages Finished at $(date)"
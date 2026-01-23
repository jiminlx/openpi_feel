STAGE2_EXP_NAME="decoupled_stream_gripper_tactile_stage2"
TOTAL_STEPS=30000  # 10k(Stage1) + 20k(Add) = 30k Total
STAGE2_TACTILE_WEIGHT=0.05 # Joint 학습 시 가중치 조절
STAGE1_STEPS=400
CHECKPOINT_DIR="/sjw_alinlab2/home/jimin/openpi_feel/checkpoints"
CONFIG_NAME="decoupled_stream_gripper_tactile"
STAGE1_EXP_NAME="decoupled_stream_gripper_tactile_stage1"

INT_STAGE1_CKPT_PATH=$((STAGE1_STEPS-1))

# Stage 1에서 저장된 마지막 체크포인트 경로 (10000 step)
STAGE1_CKPT_PATH="$CHECKPOINT_DIR/$CONFIG_NAME/$STAGE1_EXP_NAME/$INT_STAGE1_CKPT_PATH"

echo $STAGE1_CKPT_PATH
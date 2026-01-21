#!/bin/bash
#SBATCH --job-name=pi05_base          
#SBATCH --partition=sjw_alinlab
#SBATCH --nodes=1                     
#SBATCH --gpus=2                
#SBATCH --output=slurm_train_logs/%x_%j.out       # 표준 출력 로그 저장 위치 (%x: job name, %j: job id)
#SBATCH --error=slurm_train_logs/%x_%j.err        # 에러 로그 저장 위치

# To see more options, "TrainConfig" in openpi_feel/src/openpi/training/config.py
BATCH_SIZE=8
NUM_TRAIN_STEPS=60000
SAVE_INTERVAL=10000
CHECKPOINT_DIR=/sjw_alinlab2/home/jimin/openpi_feel/checkpoints
#FREEZE_FILTER=

# 로그 폴더가 없으면 에러가 날 수 있으므로 생성
mkdir -p slurm_train_logs

# uv run 명령어로 스크립트 실행
uv run scripts/train_pytorch.py naive_gripper_tactile \
    --exp_name pi05_base \
    --batch-size $BATCH_SIZE \
    --save_interval $SAVE_INTERVAL \
    --num_train_steps $NUM_TRAIN_STEPS \
    --checkpoint-base-dir $CHECKPOINT_DIR \

echo "Job finished at $(date)"
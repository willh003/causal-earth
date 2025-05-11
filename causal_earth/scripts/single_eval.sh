#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Calculate the project root directory (assuming script is in scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Configuration
PRETRAINED_MAE_PATH="/home/wph52/causal-earth/causal_earth/ckpts/fmow_pretrain.pth"
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=1e-4
OUTPUT_DIR="$PROJECT_ROOT/train_logs"
WANDB_PROJECT="earth"
ENABLE_WANDB=false

# Optional: Set additional parameters
NUM_WORKERS=8
USE_AMP=true
OPTIMIZER="adamw"
SCHEDULER="cosine"
LOG_INTERVAL=20
CHECKPOINT_INTERVAL=5
GPU_ID=0  # Set to specific GPU ID or -1 for all available GPUs
FREEZE_ENCODER=false
USE_CLS_TOKEN=true
POOL_TYPE="cls"
TASK_NAME="m-eurosat"


# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Add project root to Python path to ensure modules can be found
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Print configuration
echo "Starting MAE classifier training with the following configuration:"
echo "=====================================================>"
echo "Pretrained MAE path: $PRETRAINED_MAE_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $OUTPUT_DIR"
echo "W&B project: $WANDB_PROJECT"
echo "Task name: $TASK_NAME"
echo "Freeze encoder: $FREEZE_ENCODER"
echo "Use CLS token: $USE_CLS_TOKEN"
echo "Pool type: $POOL_TYPE"
echo "GPU ID: $GPU_ID"
echo "=====================================================>"

# Run training script from the src directory
python "$PROJECT_ROOT/src/train_eval.py" \
  --pretrained_mae_path "$PRETRAINED_MAE_PATH" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --enable_wandb "$ENABLE_WANDB" \
  --num_workers "$NUM_WORKERS" \
  --use_amp "$USE_AMP" \
  --optimizer "$OPTIMIZER" \
  --scheduler "$SCHEDULER" \
  --log_interval "$LOG_INTERVAL" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --gpu_id "$GPU_ID" \
  --freeze_encoder "$FREEZE_ENCODER" \
  --use_cls_token "$USE_CLS_TOKEN" \
  --pool_type "$POOL_TYPE" \
  --task_name "$TASK_NAME"
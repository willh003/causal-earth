#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Calculate the project root directory (assuming script is in scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Configuration
# --- Sean
# TRAIN_DIR="/home/sean/data/earthnet2021/earthnet2021x/train/"
# VAL_DIR="/home/sean/data/earthnet2021/earthnet2021x/val/"
# FAST_ACCESS=true
# CHECKPOINT_PATH="/home/sean/proj/CausalEarth/causal_earth/ckpts/fmow_pretrain.pth"

# --- Will
TRAIN_DIR="/home/wph52/earthnet2021/earthnet2021x/train/"
VAL_DIR="/home/wph52/earthnet2021/earthnet2021x/val/"
FAST_ACCESS=true
CHECKPOINT_PATH="/home/wph52/causal-earth/causal_earth/ckpts/fmow_pretrain.pth"

BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=3e-4
OUTPUT_DIR="$PROJECT_ROOT/train_logs"
WANDB_PROJECT="earthnet"
ENABLE_WANDB=false
MASK_LOSS=false

# Optional: Set additional parameters
MASK_RATIO=0.75
NUM_WORKERS=8
USE_AMP=false
OPTIMIZER="adamw"
SCHEDULER="cosine"
LOG_INTERVAL=20
CHECKPOINT_INTERVAL=5
GPU_ID=0  # Set to specific GPU ID or -1 for all available GPUs

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Add project root to Python path to ensure modules can be found
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Print configuration
echo "Starting MAE training with the following configuration:"
echo "=====================================================>"
echo "Train directory: $TRAIN_DIR"
echo "Validation directory: $VAL_DIR"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $OUTPUT_DIR"
echo "W&B project: $WANDB_PROJECT"
echo "Mask ratio: $MASK_RATIO"
echo "GPU ID: $GPU_ID"
echo "=====================================================>"

# Run training script from the src directory
python "$PROJECT_ROOT/src/train_mae_swap.py" \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --allow_fastaccess "$FAST_ACCESS" \
  --ckpt_path "$CHECKPOINT_PATH" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --enable_wandb "$ENABLE_WANDB" \
  --mask_ratio "$MASK_RATIO" \
  --num_workers "$NUM_WORKERS" \
  --use_amp "$USE_AMP" \
  --optimizer "$OPTIMIZER" \
  --scheduler "$SCHEDULER" \
  --log_interval "$LOG_INTERVAL" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --gpu_id "$GPU_ID" \
  --mask_loss "$MASK_LOSS"

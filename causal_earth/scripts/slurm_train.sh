#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Calculate the project root directory (assuming script is in scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Configuration
TRAIN_DIR="/home/wph52/earthnet2021/earthnet2021x/train/"
VAL_DIR="/home/wph52/earthnet2021/earthnet2021x/val/"
FAST_ACCESS=true
BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=3e-4
OUTPUT_DIR="$PROJECT_ROOT/train_logs"
WANDB_PROJECT="earthnet"
ENABLE_WANDB=true
MASK_LOSS=true

# Optional: Set additional parameters
NUM_WORKERS=8
USE_AMP=false
OPTIMIZER="adamw"
SCHEDULER="cosine"
LOG_INTERVAL=20
CHECKPOINT_INTERVAL=5
GPU_ID=0  # Set to specific GPU ID or -1 for all available GPUs

# Job Parameters
PARTITION="gpu"
CPUS=8
GPUS=1
MEMORY=100GB
TIME="30:00:00"

# Define multiple checkpoint paths to run
declare -a CHECKPOINT_PATHS=(
  "/home/wph52/causal-earth/causal_earth/ckpts/fmow_pretrain.pth"
  #""  # Empty string for training from scratch
)

# Define different mask ratios to experiment with
declare -a MASK_RATIOS=(
  0.75
  0.75
  #0.00 
)

# Create directories for logs and outputs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/slurm_logs"

# Loop through checkpoint paths and mask ratios
for ckpt_path in "${CHECKPOINT_PATHS[@]}"; do
  for mask_ratio in "${MASK_RATIOS[@]}"; do
    # Generate a descriptive name for the job
    if [ -z "$ckpt_path" ]; then
      ckpt_name="scratch"
    else
      ckpt_name=$(basename "$ckpt_path" .pth)
    fi
    
    job_name="mae_${ckpt_name}_mask${mask_ratio}"
    
    # Submit job to SLURM
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=${PROJECT_ROOT}/slurm_logs/${job_name}_%j.out
#SBATCH --error=${PROJECT_ROOT}/slurm_logs/${job_name}_%j.err

# Print job information
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Start time: \$(date)"
echo "=====================================================>"

# Add project root to Python path to ensure modules can be found
export PYTHONPATH="$PROJECT_ROOT:\$PYTHONPATH"

# Print configuration
echo "Starting MAE training with the following configuration:"
echo "=====================================================>"
echo "Train directory: $TRAIN_DIR"
echo "Validation directory: $VAL_DIR"
echo "Checkpoint path: ${ckpt_path}"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $OUTPUT_DIR/${job_name}"
echo "W&B project: $WANDB_PROJECT"
echo "Mask ratio: ${mask_ratio}"
echo "Mask loss: $MASK_LOSS"
echo "GPU ID: $GPU_ID"
echo "=====================================================>"

# Create run-specific output directory
mkdir -p "$OUTPUT_DIR/${job_name}"

# Run training script
python "$PROJECT_ROOT/src/train_mae.py" \\
  --train_dir "$TRAIN_DIR" \\
  --val_dir "$VAL_DIR" \\
  --allow_fastaccess "$FAST_ACCESS" \\
  --ckpt_path "${ckpt_path}" \\
  --batch_size "$BATCH_SIZE" \\
  --epochs "$EPOCHS" \\
  --learning_rate "$LEARNING_RATE" \\
  --output_dir "$OUTPUT_DIR/${job_name}" \\
  --wandb_project "$WANDB_PROJECT" \\
  --enable_wandb "$ENABLE_WANDB" \\
  --mask_ratio "${mask_ratio}" \\
  --num_workers "$NUM_WORKERS" \\
  --use_amp "$USE_AMP" \\
  --optimizer "$OPTIMIZER" \\
  --scheduler "$SCHEDULER" \\
  --log_interval "$LOG_INTERVAL" \\
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \\
  --gpu_id "$GPU_ID" \\
  --mask_loss "$MASK_LOSS"

echo "Job completed at: \$(date)"
echo "=====================================================>"
EOF

    # Print job submission information
    echo "Submitted job: ${job_name}"
    
    # Wait a bit between submissions to avoid overwhelming the scheduler
    sleep 1
  done
done
#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Calculate the project root directory (assuming script is in scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Configuration
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=1e-4
OUTPUT_DIR="$PROJECT_ROOT/train_logs"
WANDB_PROJECT="earth"
ENABLE_WANDB=true

# Optional: Set additional parameters
NUM_WORKERS=8
USE_AMP=true
OPTIMIZER="adamw"
SCHEDULER="cosine"
LOG_INTERVAL=20
CHECKPOINT_INTERVAL=5
GPU_ID=0  # Set to specific GPU ID or -1 for all available GPUs
FREEZE_ENCODER=true
USE_CLS_TOKEN=true
POOL_TYPE="cls"

# Job Parameters
PARTITION="gpu"
CPUS=8
GPUS=1
MEMORY=100GB
TIME="30:00:00"

# Define multiple tasks to run
declare -a TASKS=(
   "m-brick-kiln"
   "m-eurosat"
   "m-cashew-plant"
   "m-SA-crop-type"
  # Add more tasks here as needed
)

declare -a PRETRAINED_MAE_PATH=(
  #"/home/wph52/causal-earth/causal_earth/train_logs/mae_fmow_pretrain_mask0.75/20250503_173757_bs16_lr0.0003_wd0.05_mr0.75_pqjci/ckpts/checkpoint.pth" # pretrain
  #"/home/wph52/causal-earth/causal_earth/train_logs/mae_scratch_mask0.75/20250413_154046_bs16_lr0.0003_wd0.05_mr0.75_m5osr/ckpts/final_checkpoint.pth" # scratch
  "/home/wph52/causal-earth/causal_earth/ckpts/fmow_pretrain.pth"
  #""  # Empty string represents None
  # Add more paths here as needed
)

# Create directories for logs and outputs
if ! mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/slurm_logs"; then
    echo "Error: Failed to create required directories"
    exit 1
fi

# Loop through tasks
for task in "${TASKS[@]}"; do
  for base_path in "${PRETRAINED_MAE_PATH[@]}"; do
    # Generate a descriptive name for the job
    if [ -z "$base_path" ]; then
        model_name="scratch"
    else
        # Check if pretrained model exists
        if [ ! -f "$base_path" ]; then
            echo "Warning: Pretrained model not found at $base_path"
            continue
        fi
        model_name=$(basename "$(dirname "$(dirname "$base_path")")")
    fi
    
    job_name="classifier_${task}_${model_name}"
    
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
echo "Starting MAE classifier training with the following configuration:"
echo "=====================================================>"
if [ -z "$base_path" ]; then
    echo "Pretrained MAE path: None (training from scratch)"
else
    echo "Pretrained MAE path: $base_path"
fi
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $OUTPUT_DIR/${job_name}"
echo "W&B project: $WANDB_PROJECT"
echo "Task name: $task"
echo "Freeze encoder: $FREEZE_ENCODER"
echo "Use CLS token: $USE_CLS_TOKEN"
echo "Pool type: $POOL_TYPE"
echo "GPU ID: $GPU_ID"
echo "=====================================================>"

# Create run-specific output directory
if ! mkdir -p "$OUTPUT_DIR/${job_name}"; then
    echo "Error: Failed to create output directory"
    exit 1
fi

# Run training script
python "$PROJECT_ROOT/src/train_eval.py" \\
  --batch_size "$BATCH_SIZE" \\
  --epochs "$EPOCHS" \\
  --learning_rate "$LEARNING_RATE" \\
  --output_dir "$OUTPUT_DIR/${job_name}" \\
  --wandb_project "$WANDB_PROJECT" \\
  --enable_wandb "$ENABLE_WANDB" \\
  --num_workers "$NUM_WORKERS" \\
  --use_amp "$USE_AMP" \\
  --optimizer "$OPTIMIZER" \\
  --scheduler "$SCHEDULER" \\
  --log_interval "$LOG_INTERVAL" \\
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \\
  --gpu_id "$GPU_ID" \\
  --freeze_encoder "$FREEZE_ENCODER" \\
  --use_cls_token "$USE_CLS_TOKEN" \\
  --pool_type "$POOL_TYPE" \\
  --task_name "$task" \\
  ${base_path:+--pretrained_mae_path "$base_path"}

echo "Job completed at: \$(date)"
echo "=====================================================>"
EOF

    # Print job submission information
    echo "Submitted job: ${job_name}"
    
    # Wait a bit between submissions to avoid overwhelming the scheduler
    sleep 1
  done 
done
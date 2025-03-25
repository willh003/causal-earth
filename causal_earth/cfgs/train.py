from dataclasses import dataclass


import os
import time
import random
import string
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MAEConfig:
    """Configuration for Masked Autoencoder (MAE) training."""
    
    # Dataset paths
    train_dir: str = field(
        default="data/imagenet/train",
        metadata={"help": "Path to training dataset directory"}
    )
    val_dir: Optional[str] = field(
        default="data/imagenet/val",
        metadata={"help": "Path to validation dataset directory (optional)"}
    )
    
    allow_fastaccess: Optional[str] = field(
        default=False,
        metadata={"help": "Whether to use fast access dataset (must be precomputed with scripts/fast_access.py)"}
    )

    # Model configuration
    ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained checkpoint (optional)"}
    )
    mask_ratio: float = field(
        default=0.75,
        metadata={"help": "Ratio of masked patches during training (0.0-1.0)"}
    )
    
    # Training parameters
    batch_size: int = field(
        default=256,
        metadata={"help": "Batch size per GPU/CPU for training and evaluation"}
    )
    epochs: int = field(
        default=400,
        metadata={"help": "Total number of training epochs"}
    )
    learning_rate: float = field(
        default=1.5e-4,
        metadata={"help": "Initial learning rate (after warmup)"}
    )
    min_lr: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate for cosine scheduler"}
    )
    weight_decay: float = field(
        default=0.05,
        metadata={"help": "Weight decay coefficient"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate gradients before performing optimization"}
    )
    clip_grad: Optional[float] = field(
        default=None, 
        metadata={"help": "Maximum gradient norm (set to None to disable)"}
    )

    # Optimizer configuration
    optimizer: str = field(
        default="adamw",
        metadata={"help": "Optimizer type (adamw, sgd)"}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.95, 
        metadata={"help": "Beta2 for AdamW optimizer"}
    )
    momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum for SGD optimizer"}
    )
    
    # Scheduler configuration
    scheduler: str = field(
        default="cosine",
        metadata={"help": "LR scheduler type (cosine, step, none)"}
    )
    lr_step_size: int = field(
        default=30,
        metadata={"help": "Step size for StepLR scheduler"}
    )
    lr_gamma: float = field(
        default=0.1,
        metadata={"help": "Decay factor for StepLR scheduler"}
    )
    
    # Hardware configuration
    cuda: bool = field(
        default=True,
        metadata={"help": "Use CUDA if available"}
    )
    gpu_id: int = field(
        default=0,
        metadata={"help": "Specific GPU ID to use (-1 for all available GPUs)"}
    )
    num_workers: int = field(
        default=10,
        metadata={"help": "Number of data loading workers"}
    )
    use_amp: bool = field(
        default=True,
        metadata={"help": "Use Automatic Mixed Precision (AMP) training"}
    )
    
    # Logging and checkpointing
    output_dir: str = field(
        default="outputs/mae",
        metadata={"help": "Directory to save model checkpoints and logs"}
    )
    log_interval: int = field(
        default=10,
        metadata={"help": "Log training metrics every N steps"}
    )
    checkpoint_interval: int = field(
        default=10,
        metadata={"help": "Save checkpoint every N epochs"}
    )
    resume: bool = field(
        default=False,
        metadata={"help": "Resume training from latest checkpoint in output_dir"}
    )
    
    # Weights & Biases configuration
    wandb_project: str = field(
        default="mae-training",
        metadata={"help": "W&B project name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "W&B run name (defaults to a generated name)"}
    )
    enable_wandb: bool = field(
        default=True,
        metadata={"help": "Enable/disable Weights & Biases logging"}
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Generate timestamp and random identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        
        # Generate hyperparameter string
        hyperparams = f"bs{self.batch_size}_lr{self.learning_rate}_wd{self.weight_decay}_mr{self.mask_ratio}"
        
        # Create run directory with timestamp and hyperparams
        self.run_dir = os.path.join(self.output_dir, f"{timestamp}_{hyperparams}_{random_id}")
        
        # Create subdirectories for logs, checkpoints, and wandb
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.ckpts_dir = os.path.join(self.run_dir, "ckpts")
        self.wandb_dir = os.path.join(self.run_dir, "wandb")
        
        # Create all directories
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.ckpts_dir, exist_ok=True)
        os.makedirs(self.wandb_dir, exist_ok=True)
        
        # Set default wandb run name with timestamp and random ID
        if not self.wandb_run_name:
            self.wandb_run_name = f"mae_{timestamp}_{random_id}"
        
        # Validate parameters
        assert 0.0 <= self.mask_ratio <= 1.0, "Mask ratio must be between 0 and 1"
        assert self.optimizer in ["adamw", "sgd"], "Optimizer must be one of [adamw, sgd]"
        assert self.scheduler in ["cosine", "step", "none"], "Scheduler must be one of [cosine, step, none]"


# Example usage
if __name__ == "__main__":
    # Example with default values
    default_cfg = MAEConfig()
    print(f"Default config created with batch size: {default_cfg.batch_size}")
    
    # Example with custom values
    custom_cfg = MAEConfig(
        train_dir="data/custom_dataset/train",
        val_dir="data/custom_dataset/val",
        ckpt_path="pretrained/mae_vit_large.pth",
        batch_size=64,
        epochs=100,
        learning_rate=3e-4,
        output_dir="outputs/mae_custom",
        wandb_project="mae-custom-experiment"
    )
    print(f"Custom config created with batch size: {custom_cfg.batch_size}")


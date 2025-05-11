from dataclasses import dataclass
import yaml

import os
import time
import random
import string
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

EUROSAT_TASK_TO_NUM_CLASSES = {"m-brick-kiln": 7, "m-bigearthnet": 43, "m-eurosat": 10, "m-cashew-plant": 7, "m-SA-crop-type": 10}
EUROSAT_TASK_TO_EVAL_TYPE = {"m-brick-kiln": "classification", "m-bigearthnet": "classification", "m-eurosat": "classification", "m-cashew-plant": "segmentation", "m-SA-crop-type": "segmentation"}
EUROSAT_TASK_TO_TARGET_SIZE = {"m-SA-crop-type": (256, 256), "m-cashew-plant": (256, 256)}


@dataclass
class MAEConfig:
    """Configuration for Masked Autoencoder (MAE) or I-JEPA training."""
    
    # Model type
    model_type: str = field(default="mae", metadata={"help": "Type of model to train ('mae' or 'ijepa')"})
    
    # Dataset paths
    train_dir: str = field(default="data/imagenet/train", metadata={"help": "Path to training dataset directory"})
    mask_loss: bool = field(default=True, metadata={"help": "True to compute loss only on masked patches, false to compute on all patches"})
    val_dir: Optional[str] = field(default="data/imagenet/val", metadata={"help": "Path to validation dataset directory (optional)"})
    allow_fastaccess: bool = field(default=False, metadata={"help": "Whether to use fast access dataset (must be precomputed with scripts/fast_access.py)"})

    # Model configuration
    ckpt_path: Optional[str] = field(default=None, metadata={"help": "Path to pretrained checkpoint (optional)"})
    mask_ratio: float = field(default=0.75, metadata={"help": "Ratio of masked patches during training (0.0-1.0)"})

    # Training parameters
    batch_size: int = field(default=256, metadata={"help": "Batch size per GPU/CPU for training and evaluation"})
    epochs: int = field(default=400, metadata={"help": "Total number of training epochs"})
    learning_rate: float = field(default=1.5e-4, metadata={"help": "Initial learning rate (after warmup)"})
    min_lr: float = field(default=0.0, metadata={"help": "Minimum learning rate for cosine scheduler"})
    weight_decay: float = field(default=0.05, metadata={"help": "Weight decay coefficient"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients before performing optimization"})
    clip_grad: Optional[float] = field(default=None, metadata={"help": "Maximum gradient norm (set to None to disable)"})

    # Optimizer configuration
    optimizer: str = field(default="adamw", metadata={"help": "Optimizer type (adamw, sgd)"})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW optimizer"})
    momentum: float = field(default=0.9, metadata={"help": "Momentum for SGD optimizer"})
    
    # Scheduler configuration
    scheduler: str = field(default="cosine", metadata={"help": "LR scheduler type (cosine, step, none)"})
    lr_step_size: int = field(default=30, metadata={"help": "Step size for StepLR scheduler"})
    lr_gamma: float = field(default=0.1, metadata={"help": "Decay factor for StepLR scheduler"})
    
    # Hardware configuration
    cuda: bool = field(default=True, metadata={"help": "Use CUDA if available"})
    gpu_id: int = field(default=0, metadata={"help": "Specific GPU ID to use (-1 for all available GPUs)"})
    num_workers: int = field(default=10, metadata={"help": "Number of data loading workers"})
    use_amp: bool = field(default=True, metadata={"help": "Use Automatic Mixed Precision (AMP) training"})
    
    # Logging and checkpointing
    output_dir: str = field(default="outputs/mae", metadata={"help": "Directory to save model checkpoints and logs"})
    log_interval: int = field(default=10, metadata={"help": "Log training metrics every N steps"})
    checkpoint_interval: int = field(default=1000, metadata={"help": "Save checkpoint every N epochs"})
    resume: bool = field(default=False, metadata={"help": "Resume training from latest checkpoint in output_dir"})
    
    # Weights & Biases configuration
    wandb_project: str = field(default="mae-training", metadata={"help": "W&B project name"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "W&B run name (defaults to a generated name)"})
    wandb_tags: Optional[List[str]] = field(default=None, metadata={"help": "WB tags"})
    enable_wandb: bool = field(default=True, metadata={"help": "Enable/disable Weights & Biases logging"})
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Generate timestamp and random identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        
        # Generate hyperparameter string
        if self.model_type == "mae":
            hyperparams = f"bs{self.batch_size}_lr{self.learning_rate}_wd{self.weight_decay}_mr{self.mask_ratio}"
        else:  # ijepa
            hyperparams = f"bs{self.batch_size}_lr{self.learning_rate}_wd{self.weight_decay}_pred{self.predictor_embed_dim}"
        
        # Create run directory with timestamp and hyperparams
        self.run_dir = os.path.join(self.output_dir, f"{timestamp}_{hyperparams}_{random_id}")
        
        # Create subdirectories for logs, checkpoints, and wandb
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.cfg_dir = os.path.join(self.run_dir, "cfg")
        self.ckpts_dir = os.path.join(self.run_dir, "ckpts")
        self.wandb_dir = os.path.join(self.run_dir, "wandb")
        
        # Create all directories
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.ckpts_dir, exist_ok=True)
        os.makedirs(self.wandb_dir, exist_ok=True)
        os.makedirs(self.cfg_dir, exist_ok=True)
        
        # Set default wandb run name with timestamp and random ID
        if not self.wandb_run_name:
            self.wandb_run_name = f"{self.model_type}_{timestamp}_{random_id}"

        if self.wandb_tags is None:
            self.wandb_tags = []
        self.wandb_tags = self.wandb_tags + [
            ["all_loss", "mask_loss"][self.mask_loss],
            ["scratch", "pretrained"][bool(self.ckpt_path)],
            f"r_{self.mask_ratio}",
            self.model_type
        ]

        # Validate parameters
        assert self.model_type in ["mae", "ijepa"], "Model type must be either 'mae' or 'ijepa'"
        assert 0.0 <= self.mask_ratio <= 1.0, "Mask ratio must be between 0 and 1"
        assert self.optimizer in ["adamw", "sgd"], "Optimizer must be one of [adamw, sgd]"
        assert self.scheduler in ["cosine", "step", "none"], "Scheduler must be one of [cosine, step, none]"

        # Save config as a yaml
        with open(os.path.join(self.cfg_dir, "config.yaml"), "w") as f:
            yaml.dump(self.__dict__, f)


@dataclass
class EvalConfig(MAEConfig):
    """Configuration for training the MAE classifier"""
    pretrained_mae_path: str = None  # Path to pretrained MAE weights
    benchmark_dir: str = '/home/wph52/dataset/geobench'  # Path to geobench directory
    
    freeze_encoder: bool = True  # Whether to freeze the encoder weights
    use_cls_token: bool = True  # Whether to use the CLS token
    pool_type: str = 'cls'  # Type of pooling ('cls', 'mean', 'max')
    task_name: str = 'm-brick-kiln'  # Name of the geobench task
    output_dir: str = None  # Directory to save checkpoints and logs
    wandb_project: str = "earth"  # Weights & Biases project name
    enable_wandb: bool = True  # Whether to enable Weights & Biases logging
    num_workers: int = 8  # Number of workers for data loading
    use_amp: bool = True  # Whether to use automatic mixed precision
    optimizer: str = "adamw"  # Optimizer to use
    scheduler: str = "cosine"  # Learning rate scheduler
    log_interval: int = 20  # Interval for logging metrics
    checkpoint_interval: int = 5  # Interval for saving checkpoints
    gpu_id: int = 0  # GPU ID to use (-1 for all available GPUs)

    def __post_init__(self):
        super().__post_init__()
        self.wandb_tags = [f"{self.task_name}", "eval"]
        if self.pretrained_mae_path is not None and self.pretrained_mae_path != "":
            self.wandb_tags.append("representation")
        if self.freeze_encoder:
            self.wandb_tags.append("freeze_backbone")
        else:
            self.wandb_tags.append("full_finetune")

        self.eval_type = EUROSAT_TASK_TO_EVAL_TYPE[self.task_name]

        if self.eval_type == "segmentation":
            self.target_size = EUROSAT_TASK_TO_TARGET_SIZE[self.task_name]
        
        assert self.task_name in EUROSAT_TASK_TO_NUM_CLASSES, f"Task {self.task_name} not found in task_to_num_classes"
        self.num_classes = EUROSAT_TASK_TO_NUM_CLASSES[self.task_name]  # Number of classes for m-brick-kiln (presence/absence)

        # Save config as a yaml
        cfg_path = os.path.join(self.cfg_dir, "config.yaml")
        if os.path.exists(cfg_path):
            # remove the file if it was created by super().__post_init__
            os.remove(cfg_path)

        with open(cfg_path, "w") as f:
            yaml.dump(self.__dict__, f)

@dataclass
class IJEPAConfig(MAEConfig):
    """Configuration for I-JEPA training."""
    
        
    # I-JEPA specific parameters
    predictor_embed_dim: int = field(default=512, metadata={"help": "Embedding dimension for I-JEPA predictor"})
    predictor_depth: int = field(default=8, metadata={"help": "Number of transformer layers in I-JEPA predictor"})
    predictor_num_heads: int = field(default=16, metadata={"help": "Number of attention heads in I-JEPA predictor"})
    num_target_blocks: int = field(default=4, metadata={"help": "Number of target blocks for I-JEPA"})
    target_scale_range: tuple = field(default=(0.15, 0.2), metadata={"help": "Scale range for target blocks in I-JEPA"})
    target_aspect_ratio_range: tuple = field(default=(0.75, 1.5), metadata={"help": "Aspect ratio range for target blocks in I-JEPA"})
    context_scale_range: tuple = field(default=(0.85, 1.0), metadata={"help": "Scale range for context block in I-JEPA"})
    
    def __post_init__(self):
        super().__post_init__()

        self.wandb_tags = ["ijepa"]
            
        assert self.predictor_embed_dim > 0, "Predictor embedding dimension must be positive"
        assert self.predictor_depth > 0, "Predictor depth must be positive"
        assert self.predictor_num_heads > 0, "Predictor number of heads must be positive"
        assert self.num_target_blocks > 0, "Number of target blocks must be positive"
        assert 0 < self.target_scale_range[0] <= self.target_scale_range[1] < 1, "Target scale range must be between 0 and 1"
        assert 0 < self.target_aspect_ratio_range[0] <= self.target_aspect_ratio_range[1], "Target aspect ratio range must be positive"
        assert 0 < self.context_scale_range[0] <= self.context_scale_range[1] <= 1, "Context scale range must be between 0 and 1"


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

__all__ = ['MAEConfig', 'EvalConfig']


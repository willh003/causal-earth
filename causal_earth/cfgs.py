from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class IJEPAConfig:
    """Configuration for I-JEPA training."""
    
    # Data parameters
    train_dir: str = "data/train"
    val_dir: Optional[str] = None
    allow_fastaccess: bool = True
    
    # Model parameters
    predictor_embed_dim: int = 512
    predictor_depth: int = 8
    predictor_num_heads: int = 16
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    optimizer: str = "adamw"  # "adamw" or "sgd"
    scheduler: str = "cosine"  # "cosine", "step", or "none"
    min_lr: float = 1e-6
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    momentum: float = 0.9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    
    # Hardware parameters
    cuda: bool = True
    gpu_id: int = 0
    num_workers: int = 4
    use_amp: bool = True
    
    # Checkpointing
    ckpts_dir: str = "checkpoints"
    ckpt_path: Optional[str] = None
    resume: bool = False
    checkpoint_interval: int = 1
    
    # Logging
    enable_wandb: bool = True
    wandb_run_name: str = "ijepa"
    wandb_dir: str = "wandb"
    wandb_tags: List[str] = field(default_factory=list)
    log_interval: int = 100 
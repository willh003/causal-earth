import draccus
import wandb
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from torchvision import transforms
import torch.nn.functional as F
import geobench
from causal_earth.models.classify import create_mae_classifier
from causal_earth.models.segment import create_mae_segmentation
from causal_earth.cfgs import EvalConfig
from causal_earth.data.geobench import GeobenchCollator

from utils.train_utils import pixel_swap


def main(training_run_dir: str):
    """
    Main testing function for MAE classifier/segmentation model on geobench dataset.
    
    Args:
        cfg: Configuration object containing testing parameters
    """

    # Load config
    cfg_path = os.path.join(training_run_dir, "cfg", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    with open(os.path.join(training_run_dir, "cfg", "config.yaml"), "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    # Get valid fields from EvalConfig
    valid_fields = set(EvalConfig.__annotations__.keys())
    
    # Filter out unexpected fields
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    
    # Convert dict to EvalConfig object
    cfg = EvalConfig(**cfg_dict)

    # Initialize wandb
    initialize_wandb(cfg)
    
    # Set up device
    device = setup_device(cfg)
    
    # Build model
    model = build_model(cfg)

    # Load checkpoint
    load_checkpoint(model, cfg.checkpoint_path)
    
    # Prepare test dataloader
    test_loader = prepare_test_dataloader(cfg)
    
    # Run testing
    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        cfg=cfg
    )
    
    # Log final metrics
    log_metrics(test_metrics, prefix="test")
    
    # Close wandb
    if wandb.run:
        wandb.finish()

def initialize_wandb(cfg):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project="earth",
        entity="pcgc",
        name=cfg.wandb_run_name,
        config=vars(cfg),
        dir=cfg.wandb_dir,
        tags=cfg.wandb_tags,
        mode="online" if cfg.enable_wandb else "disabled"
    )

def setup_device(cfg):
    """Set up and return the appropriate device based on configuration."""
    if cfg.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.gpu_id}" if cfg.gpu_id >= 0 else "cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    return device

def build_model(cfg):
    """Build and initialize the model."""
    if cfg.eval_type == "classification":
        model = create_mae_classifier(
            pretrained_mae_path=cfg.pretrained_mae_path,
            num_classes=cfg.num_classes,
            freeze_encoder=cfg.freeze_encoder,
            use_cls_token=cfg.use_cls_token,
            pool_type=cfg.pool_type
        )
    elif cfg.eval_type == "segmentation":
        model = create_mae_segmentation(
            pretrained_mae_path=cfg.pretrained_mae_path,
            num_classes=cfg.num_classes,
            freeze_encoder=cfg.freeze_encoder,
            target_size=cfg.target_size
        )
        
    if cfg.use_amp:
        model = model.to(memory_format=torch.channels_last)
    
    return model

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from {checkpoint_path}")

def prepare_test_dataloader(cfg):
    """Prepare test dataloader."""
    # Get the task
    task = None
    assert cfg.eval_type in ["classification", "segmentation"], "Invalid evaluation type"
    benchmark_name = "classification_v1.0" if cfg.eval_type == "classification" else "segmentation_v1.0"
    
    for benchmark_task in geobench.task_iterator(benchmark_name=benchmark_name):
        print(benchmark_task.dataset_name)
        if benchmark_task.dataset_name == cfg.task_name:
            task = benchmark_task
            break
    
    if task is None:
        raise ValueError(f"Task {cfg.task_name} not found in geobench")
    
    # Get test dataset
    test_dataset = task.get_dataset(split="test")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create collator
    collator = GeobenchCollator(transform=transform)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator
    )
    
    return test_loader

def evaluate_model(model, data_loader, device, cfg):
    """Evaluate the model on test set."""
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = AverageMeter('Loss')
    accuracies = AverageMeter('Accuracy')
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Testing')
        
        for images, targets in pbar:
            images = images.to(device)
            images = pixel_swap(images, patch_size=16)
            targets = targets.to(device)
            
            if cfg.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            accuracy = (outputs.argmax(dim=1) == targets).float().mean()
            accuracies.update(accuracy.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses.avg,
                'acc': accuracies.avg
            })
    
    return {
        'loss': losses.avg,
        'accuracy': accuracies.avg
    }

def log_metrics(metrics, prefix="test"):
    """Log metrics to wandb."""
    if wandb.run:
        wandb.log({
            f"{prefix}/{k}": v for k, v in metrics.items()
        })

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == "__main__":
    training_run_dir = "/home/wph52/causal-earth/causal_earth/train_logs/20250510_195846_bs16_lr0.0003_wd0.05_mr0.0_ut3p8"
    main(training_run_dir) 
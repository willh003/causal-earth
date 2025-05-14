import draccus
import wandb
import torch
import numpy as np
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

@draccus.wrap()
def main(cfg: EvalConfig):
    """
    Main training function for MAE classifier on geobench dataset.
    
    Args:
        cfg: Configuration object containing training parameters
    """
    # Initialize wandb
    initialize_wandb(cfg)
    
    # Set up device
    device = setup_device(cfg)
    
    # Build model
    model = build_model(cfg)
    
    # Prepare datasets and dataloaders
    train_loader, val_loader = prepare_dataloaders(cfg)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(cfg, model)
    
    # Set up checkpointing
    checkpoint_manager = CheckpointManager(cfg.ckpts_dir)
    
    # Training loop
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        checkpoint_manager=checkpoint_manager,
    )
    
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
    """Build and initialize the classifier model."""

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

def prepare_dataloaders(cfg):
    """Prepare training and validation dataloaders."""
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
    
    # Get datasets
    train_dataset = task.get_dataset(split="train")
    val_dataset = task.get_dataset(split="valid")
    
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator
    )
    
    return train_loader, val_loader

def create_optimizer_and_scheduler(cfg, model):
    """Create optimizer and learning rate scheduler."""
    model = model.to(setup_device(cfg))
    
    # Prepare optimizer
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.adam_beta1, cfg.adam_beta2)
        )
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
    
    # Learning rate scheduler
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.epochs,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.lr_step_size,
            gamma=cfg.lr_gamma
        )
    elif cfg.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")
    
    return optimizer, scheduler

class CheckpointManager:
    """Manages model checkpointing."""
    
    def __init__(self, ckpts_dir):
        self.ckpts_dir = ckpts_dir
        os.makedirs(ckpts_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        """Save model checkpoint."""
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        
        if scheduler:
            checkpoint['scheduler'] = scheduler.state_dict()
        
        filename = "checkpoint.pth"
        checkpoint_path = os.path.join(self.ckpts_dir, filename)
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, cfg, checkpoint_manager):
    """Main training loop."""
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if cfg.use_amp else None
    
    for epoch in range(cfg.epochs):
        # Training
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
            cfg=cfg
        )
        
        # Validation
        if val_loader:
            val_metrics = evaluate_model(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                cfg=cfg
            )
            log_metrics(val_metrics, epoch=epoch, prefix="val")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, epoch)
        
        # Log metrics
        log_metrics(train_metrics, epoch=epoch, prefix="train")


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, scaler, cfg):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter('Loss')
    accuracies = AverageMeter('Accuracy')
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        images = pixel_swap(images, patch_size=16)
        targets = targets.to(device)
        
        # Forward pass
        if cfg.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        if cfg.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
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

def evaluate_model(model, data_loader, criterion, device, epoch, cfg):
    """Evaluate the model."""
    model.eval()
    losses = AverageMeter('Loss')
    accuracies = AverageMeter('Accuracy')
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'Epoch {epoch} [Val]')
        
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

def log_metrics(metrics, epoch, prefix="train"):
    """Log metrics to wandb."""
    if wandb.run:
        wandb.log({
            f"{prefix}/{k}": v for k, v in metrics.items()
        }, step=epoch)

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
    main() 
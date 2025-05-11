import draccus
import glob
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from typing import Union

import earthnet_models_pytorch as emp
from earthnet_models_pytorch.data.en21x_data import EarthNet2021XDataset

from causal_earth.models import ijepa_vit_large_patch16
from causal_earth.cfgs.train import MAEConfig
from causal_earth.utils import interpolate_pos_embed
from causal_earth.data.earthnet import IJEPACollator

@draccus.wrap()
def main(cfg: MAEConfig):
    """
    Main training function for I-JEPA.
    
    Args:
        cfg: Configuration object containing training parameters
    """
    # Set model type to ijepa
    cfg.model_type = "ijepa"
    
    # Initialize wandb
    initialize_wandb(cfg)
    
    # Set up device
    device = setup_device(cfg)
    
    # Build model
    model = build_model(cfg)
    
    # Create transform pipeline
    transform = transforms.Compose([
        # Resize the image to 224x224 pixels
        transforms.Lambda(lambda x: F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create collator
    collator = IJEPACollator(
        transform=transform,
        num_target_blocks=cfg.num_target_blocks,
        patch_size=16  # Fixed for ViT-Large
    )
    
    # Prepare datasets and dataloaders
    train_set = EarthNet2021XDataset(cfg.train_dir, dl_cloudmask = True, allow_fastaccess = cfg.allow_fastaccess)
    val_set = EarthNet2021XDataset(cfg.val_dir, dl_cloudmask = True, allow_fastaccess = cfg.allow_fastaccess) if cfg.val_dir else None
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator
    ) if val_set else None

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
    
    # Final evaluation
    if val_loader:
        evaluate_model(model, val_loader, device, epoch="final", cfg=cfg)
        log_metrics(val_metrics, epoch=0, prefix="val")
    
    # Save final model
    checkpoint_manager.save_checkpoint(model, optimizer, scheduler, cfg.epochs)
    
    # Close wandb
    if wandb.run:
        wandb.finish()


def initialize_wandb(cfg):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=cfg.wandb_project,
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
    
    # Log device info
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    return device


def build_model(cfg):
    """Build and initialize the model with pretrained weights if available."""
    # Initialize model
    model = ijepa_vit_large_patch16(
        predictor_embed_dim=cfg.predictor_embed_dim,
        predictor_depth=cfg.predictor_depth,
        predictor_num_heads=cfg.predictor_num_heads
    )
    
    # Load pre-trained weights if provided
    if cfg.ckpt_path:
        model = load_pretrained_weights(model, cfg.ckpt_path)
    
    # Enable mixed precision if requested
    if cfg.use_amp:
        model = model.to(memory_format=torch.channels_last)
    
    return model


def load_pretrained_weights(model, ckpt_path):
    """Load pretrained weights into the model."""
    print(f"Loading pre-trained checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    # Handle incompatible keys
    incompatible_keys = ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']
    for k in incompatible_keys:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Handle positional embeddings
    interpolate_pos_embed(model, checkpoint_model)

    # Load state dict
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    return model


def create_optimizer_and_scheduler(cfg, model):
    """Create optimizer and learning rate scheduler."""
    # Move model to device before creating optimizer
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
        
    def load_latest_checkpoint(self, model, optimizer, scheduler=None):
        """Load the latest checkpoint if available."""
        checkpoints = glob.glob(os.path.join(self.ckpts_dir, "checkpoint_epoch_*.pth"))
        if not checkpoints:
            return 0  # No checkpoints found, start from epoch 0
        
        # Find the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r"checkpoint_epoch_(\d+).pth", x).group(1)))
        checkpoint = torch.load(latest_checkpoint)
        
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint {latest_checkpoint}, starting at epoch {start_epoch}")
        return start_epoch


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, cfg, checkpoint_manager):
    """
    Main training loop.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (can be None)
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (can be None)
        device: Device to use for training
        cfg: Configuration object
        checkpoint_manager: Checkpoint manager
    """
    # Move model to device
    model = model.to(device)

    # Initialize AMP scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if cfg.use_amp else None
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if cfg.resume:
        start_epoch = checkpoint_manager.load_latest_checkpoint(model, optimizer, scheduler)

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        print(f"Starting epoch {epoch+1}/{cfg.epochs}")
        
        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            cfg=cfg
        )
        
        # Log training metrics
        log_metrics(train_metrics, epoch, prefix="train")
        
        # Validate if we have a validation set
        if val_loader:
            val_metrics = evaluate_model(model, val_loader, device, epoch, cfg)
            log_metrics(val_metrics, epoch, prefix="val")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            if wandb.run:
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})
        
        # Save checkpoint
        if (epoch + 1) % cfg.checkpoint_interval == 0:
            checkpoint_manager.save_checkpoint(model, optimizer, scheduler, epoch)


def train_one_epoch(model, data_loader, optimizer, device, epoch, scaler, cfg):
    """Train for one epoch."""
    model.train()
    
    # Initialize metrics
    data_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4f')
    
    # Progress bar
    progress = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    end = time.time()
    for batch_idx, (images, context_masks, target_masks) in enumerate(progress):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        images = images.to(device, non_blocking=True)
        context_masks = context_masks.to(device, non_blocking=True)
        target_masks = target_masks.to(device, non_blocking=True)
        
        # Forward pass with mixed precision if enabled
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            loss = model(images, context_masks, target_masks)
        
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
        loss_meter.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        progress.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'data_time': f'{data_time.avg:.3f}s',
            'batch_time': f'{batch_time.avg:.3f}s'
        })
        
        # Log to wandb
        if wandb.run and batch_idx % cfg.log_interval == 0:
            wandb.log({
                'train/loss': loss_meter.avg,
                'train/data_time': data_time.avg,
                'train/batch_time': batch_time.avg,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/epoch': epoch,
                'train/step': epoch * len(data_loader) + batch_idx
            })
            
        # Visualize some examples
        if batch_idx == 0:
            # Get predictions for visualization
            with torch.no_grad():
                pred_images = model.predict(images, context_masks)
            
            # Visualize first image in batch
            vis_image = images[0]
            vis_pred = pred_images[0]
            vis_context = context_masks[0, 0]
            vis_target = target_masks[0].sum(dim=0) > 0
            
            # Create visualization
            example_images = {
                "full_image": wandb.Image(vis_image.cpu().permute(1, 2, 0).numpy() * 255, caption="full image"),
                "context_mask": wandb.Image(vis_context.cpu().numpy() * 255, caption="context mask"),
                "target_mask": wandb.Image(vis_target.cpu().numpy() * 255, caption="target mask"),
                "prediction": wandb.Image(vis_pred.detach().cpu().permute(1, 2, 0).numpy() * 255, caption="prediction"),
            }
            
            log_metrics(example_images, epoch, prefix="train")
    
    return {
        'loss': loss_meter.avg,
        'data_time': data_time.avg,
        'batch_time': batch_time.avg
    }


def evaluate_model(model, data_loader, device, epoch, cfg):
    """Evaluate the model."""
    model.eval()
    
    # Initialize metrics
    loss_meter = AverageMeter('Loss', ':.4f')
    
    # Progress bar
    progress = tqdm(data_loader, desc=f'Evaluation {epoch}')
    
    with torch.no_grad():
        for batch_idx, (images, context_masks, target_masks) in enumerate(progress):
            # Move data to device
            images = images.to(device, non_blocking=True)
            context_masks = context_masks.to(device, non_blocking=True)
            target_masks = target_masks.to(device, non_blocking=True)
            
            # Forward pass
            loss = model(images, context_masks, target_masks)
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            
            # Update progress bar
            progress.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
            
            # Visualize some examples
            if batch_idx == 0:
                # Get predictions for visualization
                pred_images = model.predict(images, context_masks)
                
                # Visualize first image in batch
                vis_image = images[0]
                vis_pred = pred_images[0]
                vis_context = context_masks[0, 0]
                vis_target = target_masks[0].sum(dim=0) > 0
                
                # Create visualization
                example_images = {
                    "full_image": wandb.Image(vis_image.cpu().permute(1, 2, 0).numpy() * 255, caption="full image"),
                    "context_mask": wandb.Image(vis_context.cpu().numpy() * 255, caption="context mask"),
                    "target_mask": wandb.Image(vis_target.cpu().numpy() * 255, caption="target mask"),
                    "prediction": wandb.Image(vis_pred.detach().cpu().permute(1, 2, 0).numpy() * 255, caption="prediction"),
                }
                
                log_metrics(example_images, epoch, prefix="val")
    
    return {
        'loss': loss_meter.avg
    }


def log_metrics(metrics, epoch, prefix="train"):
    """Log metrics to wandb and console."""
    # Log to wandb
    if wandb.run:
        wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb_metrics['epoch'] = epoch
        wandb.log(wandb_metrics)


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
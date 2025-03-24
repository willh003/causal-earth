
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

import earthnet_models_pytorch as emp
from earthnet_models_pytorch.data.en21x_data import EarthNet2021XDataset

from causal_earth.models import mae_vit_large_patch16_dec512d8b
from causal_earth.cfgs import MAEConfig
from causal_earth.utils import interpolate_pos_embed, visualize_masked_image
from causal_earth.data import create_pooled_rgb_dataset

@draccus.wrap()
def main(cfg: MAEConfig):
    """
    Main training function for Masked Autoencoder (MAE).
    
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
    train_set = EarthNet2021XDataset(cfg.train_dir, dl_cloudmask = True, allow_fastaccess = cfg.allow_fastaccess)
    val_set = EarthNet2021XDataset(cfg.val_dir, dl_cloudmask = True, allow_fastaccess = cfg.allow_fastaccess) if cfg.val_dir else None
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    ) if val_set else None

    # Create a comprehensive transform pipeline
    transform = transforms.Compose([
        # Resize the image to 224x224 pixels
        transforms.Lambda(lambda x: F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(cfg, model)
    
    # Set up checkpointing
    checkpoint_manager = CheckpointManager(cfg.ckpts_dir)
    
    # First evaluation
    if val_loader:
        val_metrics = evaluate_model(model, transform, val_loader, device, epoch=0, cfg=cfg)
        log_metrics(val_metrics, epoch=0, prefix="val")
    
    # Training loop
    train_model(
        model=model,
        transform=transform,
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
        evaluate_model(model, transform, val_loader, device, epoch="final", cfg=cfg)
    
    # Save final model
    checkpoint_manager.save_checkpoint(model, optimizer, scheduler, cfg.epochs, is_final=True)
    
    # Close wandb
    if wandb.run:
        wandb.fninish()


def initialize_wandb(cfg):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project="temporal-earth-prac",
        entity="willhu003",
        name=cfg.wandb_run_name,
        config=vars(cfg),
        dir=cfg.wandb_dir,
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
    model = mae_vit_large_patch16_dec512d8b()
    
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
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, is_final=False):
        """Save model checkpoint."""
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        
        if scheduler:
            checkpoint['scheduler'] = scheduler.state_dict()
        
        filename = "final_checkpoint.pth" if is_final else f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.ckpts_dir, filename)
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Log to wandb
        if wandb.run:
            wandb.save(checkpoint_path)
    
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


def train_model(model, transform, train_loader, val_loader, optimizer, scheduler, device, cfg, checkpoint_manager):
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
            transform=transform,
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
            val_metrics = evaluate_model(model, transform, val_loader, device, epoch, cfg)
            log_metrics(val_metrics, epoch, prefix="val")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            if wandb.run:
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})
        
        # Save checkpoint
        if (epoch + 1) % cfg.checkpoint_interval == 0:
            checkpoint_manager.save_checkpoint(model, optimizer, scheduler, epoch)


def train_one_epoch(model, transform, data_loader, optimizer, device, epoch, scaler, cfg):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_samples = 0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4f')
    
    # Create log file for this epoch
    log_file = os.path.join(cfg.logs_dir, f"train_epoch_{epoch}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    progress = tqdm(enumerate(data_loader), total=len(data_loader), 
                   desc=f"Epoch {epoch+1}/{cfg.epochs}")
    end = time.time()
    
    for batch_idx, data in progress:
        # Measure data loading time
        data_time.update(time.time() - end)

        # TODO: factor this out into dataset
        dynamic_bgr = data["dynamic"][0][:, 0, 1:4, ...] # [0] grabs the rgb and not cloud, 1:4 grabs bgr bands, : grabs batch
        dynamic_rgb = dynamic_bgr.flip(1) # flip along channel to get rgb
        images = transform(dynamic_rgb) # normalize the rgb image

        # Move data to device
        images = images.to(device, non_blocking=True)
        
        # Forward pass with mixed precision if enabled
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            loss, pred_images, mask = model(images, mask_ratio=cfg.mask_ratio)
        
        # Normalize loss by accumulation steps
        loss = loss / cfg.gradient_accumulation_steps
        
        # Backward pass with mixed precision if enabled
        if scaler:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                # Unscale gradients for potential gradient clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients if requested
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                
                # Optimizer step with scaled gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                # Clip gradients if requested
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item() * cfg.gradient_accumulation_steps  # Scale back for reporting
        num_samples += images.size(0)
        loss_meter.update(loss.item() * cfg.gradient_accumulation_steps, images.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        progress.set_postfix(loss=loss_meter.avg)
        
        if batch_idx == 0:
            patch_size = int(224/14) # TODO: hard coded patch size
            unpatched_preds = model.unpatchify(pred_images, p=patch_size, c=images.shape[1]) 

            vis_image = images[0]
            vis_pred = unpatched_preds[0]
            pixel_mask = mask[:, :, None].expand(*mask.size(), pred_images.shape[-1] // images.shape[1]) # expand from 14x14 patches to 224*224 pixels
            vis_mask = model.unpatchify(pixel_mask.int(),p=patch_size, c=1)[0] # get mask for entire image

            masked_image = visualize_masked_image(vis_image, vis_mask, patch_size=16) # patch size 16 for pretrained MAEs
            masked_preds = visualize_masked_image(vis_pred, 1-vis_mask, patch_size=16) # patch size 16 for pretrained MAEs

            metrics = {
            # wandb uses PIL format (h, w, c) * 255
            "train/full_image": wandb.Image(vis_image.cpu().permute(1, 2, 0).numpy() * 255, caption="full image"),
            "train/masked_image": wandb.Image(masked_image * 255, caption="masked image"),
            "train/masked_preds": wandb.Image(masked_preds * 255, caption="masked preds"),
            "train/full_preds": wandb.Image(vis_pred.detach().cpu().permute(1, 2, 0).numpy() * 255, caption="full preds"),
            }

            log_metrics(metrics, epoch, prefix="train")

        # Log metrics intermittently
        if batch_idx % cfg.log_interval == 0 and wandb.run:
            wandb.log({
                "train/batch_loss": loss_meter.avg,
                "train/batch_time": batch_time.avg,
                "train/data_time": data_time.avg
            })
    
    # Compute epoch metrics
    avg_loss = total_loss / len(data_loader)
    
    return {
        "loss": avg_loss,
        "batch_time": batch_time.avg,
        "data_time": data_time.avg
    }


def evaluate_model(model, transform, data_loader, device, epoch, cfg):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    # Create log file for validation
    log_file = os.path.join(cfg.logs_dir, f"val_epoch_{epoch}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with torch.no_grad():
        progress = tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation")
        
        for batch_idx, data in progress:
            # Move data to device

            dynamic_bgr = data["dynamic"][0][:, 0, 1:4, ...] # [0] grabs the rgb and not cloud, 1:4 grabs bgr bands, : grabs batch
            dynamic_rgb = dynamic_bgr.flip(1) # flip along channel to get rgb
            images = transform(dynamic_rgb) # normalize the rgb image

            images = images.to(device, non_blocking=True)
            
            # Forward pass
            loss, pred_images, mask = model(images, mask_ratio=cfg.mask_ratio)  # Fixed mask ratio for validation
            
            # Update metrics
            total_loss += loss.item()
            num_samples += images.size(0)
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            progress.set_postfix(loss=current_loss)
            
            # Save log to file
            with open(log_file, 'a') as f:
                f.write(f"Batch {batch_idx}/{len(data_loader)}, Loss: {current_loss:.4f}\n")
            
            if batch_idx == 0:
                patch_size = int(224/14) # TODO: hard coded patch size
                unpatched_preds = model.unpatchify(pred_images, p=patch_size, c=images.shape[1]) 

                vis_image = images[0]
                vis_pred = unpatched_preds[0]
                pixel_mask = mask[:, :, None].expand(*mask.size(), pred_images.shape[-1] // images.shape[1]) # expand from 14x14 patches to 224*224 pixels
                vis_mask = model.unpatchify(pixel_mask.int(),p=patch_size, c=1)[0] # get mask for entire image

                masked_image = visualize_masked_image(vis_image, vis_mask, patch_size=16) # patch size 16 for pretrained MAEs
                masked_preds = visualize_masked_image(vis_pred, 1-vis_mask, patch_size=16) # patch size 16 for pretrained MAEs

                metrics = {
                    # wandb uses PIL format (h, w, c) * 255
                    "val/full_image": wandb.Image(vis_image.cpu().permute(1, 2, 0).numpy() * 255, caption="full image"),
                    "val/masked_image": wandb.Image(masked_image * 255, caption="masked image"),
                    "val/masked_preds": wandb.Image(masked_preds * 255, caption="masked preds"),
                    "val/full_preds": wandb.Image(vis_pred.detach().cpu().permute(1, 2, 0).numpy() * 255, caption="full preds"),
                }
                log_metrics(metrics, epoch, prefix="val")

    # Compute average metrics
    avg_loss = total_loss / len(data_loader)
    
    return {
        "loss": avg_loss,
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
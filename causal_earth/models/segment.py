import torch
import torch.nn as nn
from functools import partial
from causal_earth.models import mae_vit_large_patch16_dec512d8b


class MAESegmenter(nn.Module):
    """MAE with a segmentation head for pixel-wise prediction tasks"""
    
    def __init__(self, mae_model, num_classes, target_size, freeze_encoder=True):
        """
        Args:
            mae_model: Pretrained MAE model
            num_classes: Number of output classes for segmentation
            freeze_encoder: Whether to freeze the MAE encoder weights
        """
        super().__init__()
        
        # Store the MAE encoder components
        self.patch_embed = mae_model.patch_embed
        self.cls_token = mae_model.cls_token
        self.pos_embed = mae_model.pos_embed
        self.blocks = mae_model.blocks
        self.norm = mae_model.norm
        self.target_size = target_size
        
        # Segmentation head parameters
        self.embed_dim = mae_model.blocks[0].norm1.normalized_shape[0]
        self.patch_size = mae_model.patch_embed.patch_size[0]
        
        # Create segmentation head with upsampling
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Initialize segmentation head
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze the MAE encoder parameters"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
    
    def forward_encoder(self, x):
        """Forward pass through the encoder without masking"""
        # Embed patches
        x = self.patch_embed(x)
        
        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x
    
    def forward(self, x):
        """Forward pass for segmentation"""
        # Get encoder features
        x = self.forward_encoder(x)
        
        # Remove CLS token and reshape for segmentation
        x = x[:, 1:]  # Remove CLS token
        
        # Get spatial dimensions
        B, L, C = x.shape
        H = W = int((L) ** 0.5)
        
        # Reshape to spatial format
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Apply segmentation head
        x = self.seg_head(x)
        
        # Upsample to target size (256x256)
        x = nn.functional.interpolate(
            x, 
            size=self.target_size,  # Fixed target size
            mode='bilinear',
            align_corners=False
        )

        return x
    
    def get_intermediate_layers(self, x, n=1):
        """Get intermediate layer outputs for visualization or analysis"""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                outputs.append(self.norm(x))
        
        return outputs


def create_mae_segmentation(pretrained_mae_path=None, num_classes=2, freeze_encoder=True, target_size=(256, 256)):
    """
    Create an MAE segmenter from a pretrained MAE model
    
    Args:
        pretrained_mae_path: Path to pretrained MAE weights
        num_classes: Number of output classes for segmentation
        freeze_encoder: Whether to freeze the encoder weights
    
    Returns:
        MAESegmenter model
    """
    # Create the base MAE model
    model = mae_vit_large_patch16_dec512d8b()

    # Load pretrained weights if provided
    if pretrained_mae_path is not None and pretrained_mae_path != "":
        checkpoint = torch.load(pretrained_mae_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load only encoder weights
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.') or k.startswith('patch_embed.') or \
               k.startswith('blocks.') or k == 'cls_token' or k == 'pos_embed' or \
               k.startswith('norm.'):
                # Remove 'encoder.' prefix if present
                new_k = k.replace('encoder.', '') if k.startswith('encoder.') else k
                encoder_state_dict[new_k] = v
        
        # Load pretrained weights
        model.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_mae_path}")
    
    # Create segmenter
    segmenter = MAESegmenter(model, num_classes, target_size, freeze_encoder)
    
    return segmenter


# Example usage
if __name__ == "__main__":
    # Create a segmenter
    model = create_mae_segmentation(
        pretrained_mae_path="/home/wph52/causal-earth/causal_earth/train_logs/mae_fmow_pretrain_mask0.75/20250503_173757_bs16_lr0.0003_wd0.05_mr0.75_pqjci/ckpts/checkpoint.pth",
        num_classes=2,
        freeze_encoder=True
    )
    
    # Example forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [2, num_classes, 224, 224]
    
    # Example of getting intermediate features
    intermediate_features = model.get_intermediate_layers(dummy_input, n=2)
    print(f"Number of intermediate layers: {len(intermediate_features)}")
    print(f"Last intermediate feature shape: {intermediate_features[-1].shape}") 
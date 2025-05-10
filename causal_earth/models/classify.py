import torch
import torch.nn as nn
from functools import partial
from causal_earth.models import mae_vit_large_patch16_dec512d8b


# Import the MAE class from the original file
# from causal_earth.models_mae import MAE, mae_vit_base_patch16_dec512d8b

class MAEClassifier(nn.Module):
    """MAE with a classification head for downstream tasks"""
    
    def __init__(self, mae_model, num_classes, freeze_encoder=True, use_cls_token=True, pool_type='cls'):
        """
        Args:
            mae_model: Pretrained MAE model
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze the MAE encoder weights
            use_cls_token: Whether to use the CLS token for classification
            pool_type: Type of pooling ('cls', 'mean', 'max')
        """
        super().__init__()
        
        # Store the MAE encoder components
        self.patch_embed = mae_model.patch_embed
        self.cls_token = mae_model.cls_token
        self.pos_embed = mae_model.pos_embed
        self.blocks = mae_model.blocks
        self.norm = mae_model.norm
        
        # Classification head parameters
        self.use_cls_token = use_cls_token
        self.pool_type = pool_type
        self.embed_dim = mae_model.blocks[0].norm1.normalized_shape[0]
        
        # Create classification head
        self.fc_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        # Initialize classification head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
        
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
        """Forward pass for classification"""
        # Get encoder features
        x = self.forward_encoder(x)
        
        # Pool features
        if self.use_cls_token and self.pool_type == 'cls':
            # Use CLS token
            x = x[:, 0]
        else:
            # Pool patch tokens
            x = x[:, 1:]  # Remove CLS token
            if self.pool_type == 'mean':
                x = x.mean(dim=1)
            elif self.pool_type == 'max':
                x = x.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pool type: {self.pool_type}")
        
        # Classification head
        x = self.fc_norm(x)
        x = self.head(x)
        
        # Apply softmax to get class probabilities
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


def create_mae_classifier(pretrained_mae_path=None, num_classes=1000, freeze_encoder=True, 
                         use_cls_token=True, pool_type='cls'):
    """
    Create an MAE classifier from a pretrained MAE model
    
    Args:
        pretrained_mae_path: Path to pretrained MAE weights
        num_classes: Number of output classes
        freeze_encoder: Whether to freeze the encoder weights
        use_cls_token: Whether to use the CLS token
        pool_type: Type of pooling ('cls', 'mean', 'max')
    
    Returns:
        MAEClassifier model
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
    
    # Create classifier
    classifier = MAEClassifier(model, num_classes, freeze_encoder, 
                              use_cls_token, pool_type)
    
    return classifier


# Example usage
if __name__ == "__main__":
    # Create a classifier
    model = create_mae_classifier(
        pretrained_mae_path="/home/wph52/causal-earth/causal_earth/train_logs/mae_fmow_pretrain_mask0.75/20250503_173757_bs16_lr0.0003_wd0.05_mr0.75_pqjci/ckpts/checkpoint.pth",
        num_classes=10,
        freeze_encoder=True,
        use_cls_token=True,
        pool_type='cls'
    )
    
    # Example forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [2, 10]
    
    # Example of getting intermediate features
    intermediate_features = model.get_intermediate_layers(dummy_input, n=2)
    print(f"Number of intermediate layers: {len(intermediate_features)}")
    print(f"Last intermediate feature shape: {intermediate_features[-1].shape}")
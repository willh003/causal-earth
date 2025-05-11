from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from causal_earth.utils.model_utils import get_2d_sincos_pos_embed


class IJEPA(nn.Module):
    """ Image Joint Embedding-Predictive Architecture
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 predictor_embed_dim=512, predictor_depth=8, predictor_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.in_c = in_chans
        self.patch_size = patch_size
        self.img_size = img_size

        # --------------------------------------------------------------------------
        # Target encoder specifics
        self.target_patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.target_patch_embed.num_patches

        self.target_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.target_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.target_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.target_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Context encoder specifics
        self.context_patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.context_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.context_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.context_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.context_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Predictor specifics
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, predictor_embed_dim), requires_grad=False)

        self.predictor_blocks = nn.ModuleList([
            Block(predictor_embed_dim, predictor_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(predictor_depth)])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_pred = nn.Linear(predictor_embed_dim, embed_dim, bias=True)  # predictor to target space
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.target_pos_embed.shape[-1], int(self.target_patch_embed.num_patches**.5), cls_token=True)
        self.target_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.context_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1], int(self.target_patch_embed.num_patches**.5), cls_token=True)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.target_patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.context_patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.target_cls_token, std=.02)
        torch.nn.init.normal_(self.context_cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_target_encoder(self, x, target_masks):
        # Reshape target masks to match sequence length
        b, l, h, w = target_masks.shape
        target_masks = target_masks.view(b, h*w, l)
        
        # Apply target encoder
        x = self.target_patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.target_pos_embed[:, 1:, :]
        
        # Apply masks
        x = x * target_masks  # [B, 196, 1024]
        
        # append cls token
        cls_token = self.target_cls_token + self.target_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.target_blocks:
            x = blk(x)
        x = self.target_norm(x)

        return x

    def forward_context_encoder(self, x, context_masks):
        # embed patches
        b, l, h, w = context_masks.shape
        context_masks = context_masks.view(b, h*w, l)

        x = self.context_patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.context_pos_embed[:, 1:, :]

        # Apply context masks
        x = x * context_masks

        # append cls token
        cls_token = self.context_cls_token + self.context_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.context_blocks:
            x = blk(x)
        x = self.context_norm(x)

        return x

    def forward_predictor(self, x):
        # embed tokens
        x = self.predictor_embed(x)

        # add pos embed
        x = x + self.predictor_pos_embed

        # apply Transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # predictor projection
        x = self.predictor_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, target_embeddings, pred_embeddings, target_masks):
        """
        target_embeddings: [N, L, D]
        pred_embeddings: [N, L, D]
        target_masks: [N, 4, L], 1 is keep, 0 is remove
        """
        # Normalize embeddings
        target_embeddings = F.normalize(target_embeddings, dim=-1)
        pred_embeddings = F.normalize(pred_embeddings, dim=-1)
        
        # Compute cosine similarity for each target mask
        similarity = torch.sum(target_embeddings * pred_embeddings, dim=-1)  # [N, L]
        similarity = similarity.unsqueeze(1) * target_masks  # [N, 4, L]
        
        # Average over valid positions for each mask
        loss = -similarity.sum(dim=-1) / (target_masks.sum(dim=-1) + 1e-6)  # [N, 4]
        
        return loss.mean()  # Average over batch and masks

    def forward(self, imgs, target_masks, context_masks):
        # Get target embeddings
        target_embeddings = self.forward_target_encoder(imgs, target_masks)
        
        # Get context embeddings
        context_embeddings = self.forward_context_encoder(imgs, context_masks)
        
        # Predict masked embeddings
        pred_embeddings = self.forward_predictor(context_embeddings)
        
        # Compute loss
        loss = self.forward_loss(target_embeddings[:, 1:], pred_embeddings, target_masks)
        
        return loss, pred_embeddings, target_masks


def ijepa_vit_base_patch16(**kwargs):
    model = IJEPA(
        embed_dim=768, depth=12, num_heads=12,
        predictor_embed_dim=512, predictor_depth=8, predictor_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ijepa_vit_large_patch16(**kwargs):
    model = IJEPA(**kwargs)
    return model


def ijepa_vit_huge_patch14(**kwargs):
    model = IJEPA(
        embed_dim=1280, depth=32, num_heads=16,
        predictor_embed_dim=512, predictor_depth=8, predictor_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model 
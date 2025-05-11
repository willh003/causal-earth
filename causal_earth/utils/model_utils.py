# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# SatMAE: https://github.com/sustainlab-group/SatMAE
# --------------------------------------------------------

import sys
import numpy as np
import torch
from timm.models.vision_transformer import Attention
from PIL import Image, ImageFilter


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=np.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def attention_rollout(attentions):
    # Initialize rollout with identity matrix
    rollout = torch.eye(attentions.size(-1)).to(attentions.device)
    print("Rollout shape: ", rollout.shape)
    # Multiply attention maps layer by layer
    for attention in attentions:
        attention_heads_fused = attentions.mean(dim=1)  # Average attention across heads
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(
            attention_heads_fused.device
        )  # A + I
        attention_heads_fused /= attention_heads_fused.sum(
            dim=-1, keepdim=True
        )  # Normalizing A
        rollout = torch.matmul(rollout, attention_heads_fused)  # Multiplication

    return rollout


def generate_attention_matrix(qkv_pre, transformer_input):
    qkv_0 = qkv_pre(transformer_input)[0].reshape(197, 3, 16, 64)
    q = qkv_0[:, 0].permute(1, 0, 2)
    k = qkv_0[:, 1].permute(1, 0, 2)
    kT = k.permute(0, 2, 1)

    # Attention Matrix
    attention_matrix = q @ kT
    return attention_matrix


def rgba_to_rgb(image):
    """
    Converts an RGBA image to RGB, saving the result.
    If the image is not RGBA, it returns the original image.
    """
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image


def get_attn_maps(model, x, mask=None):
    """
    Get attention maps from the model.
    Args:
        model: The model to get attention maps from.
        x: The input tensor.
        mask: The mask tensor (optional).
    Returns:
        attn_maps: The attention maps.
    """
    attentions = {"encoder": [], "decoder": []}  # Placeholder for attention maps
    # Forward pass through the model to get attention maps
    # print(model)

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if name.startswith("blocks"):
                attentions["encoder"].append(module)
            elif name.startswith("decoder"):
                attentions["decoder"].append(module)
    encoder_matrices = []

    patch_input = model.patch_embed(x)
    pos_embed = model.pos_embed
    transformer_input = torch.cat((model.cls_token, patch_input), dim=1) + pos_embed
    print("Transformer input: ", transformer_input.shape)
    # Attention Matrix
    for i in range(len(attentions["encoder"])):
        encoder_matrices.append(
            generate_attention_matrix(attentions["encoder"][i].qkv, transformer_input)
        )
    encoder_matrices = torch.stack(encoder_matrices, dim=0)
    print("Encoder matrices: ", encoder_matrices)
    np.set_printoptions(threshold=sys.maxsize)

    rollout = attention_rollout(encoder_matrices)
    cls_attention = rollout[
        0, 1:, 0
    ]  # Get attention values from [CLS] token to all patches
    print("cls_attention: ", cls_attention)
    cls_attention = 1 - cls_attention.reshape(
        int(np.sqrt(model.patch_embed.num_patches)),
        int(np.sqrt(model.patch_embed.num_patches)),
    )
    # Normalize the attention map for better visualization
    cls_attention = (cls_attention - cls_attention.min()) / (
        cls_attention.max() - cls_attention.min()
    )
    print("cls_attention normalized: ", cls_attention)
    # Resize and blur the attention map
    imsize = 128
    cls_attention_resized = Image.fromarray(
        (cls_attention * 255).to(torch.uint8).cpu().numpy()
    ).resize((imsize, imsize), resample=Image.BICUBIC)
    cls_attention_resized = cls_attention_resized.filter(
        ImageFilter.GaussianBlur(radius=2)
    )
    # Convert the attention map to RGBA
    cls_attention_colored = np.array(cls_attention_resized.convert("L"))
    cls_attention_colored = np.stack(
        [cls_attention_colored] * 3 + [cls_attention_colored], axis=-1
    )

    # Adjust the alpha channel to control brightness
    cls_attention_colored_img = Image.fromarray(cls_attention_colored, mode="RGBA")
    cls_attention_colored_img.putalpha(
        100
    )  # Adjust alpha for blending (lower value for darker overlay)
    print(
        "cls_attention_colored_img: ",
        np.asarray(rgba_to_rgb(cls_attention_colored_img)).shape,
    )
    return cls_attention_colored_img

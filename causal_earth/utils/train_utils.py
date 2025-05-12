

import torch

def pixel_swap(images, patch_dim):
    """Returns the image corrupted with a pixel swap along every vertical pair of patches.

    This function is its own inverse.

    Params:
        images : (B,C,H,W) Tensor
        patch_dim : int with below properties 

    Requires: 
        1) (width / patch_dim) % 2 == 0
        2) (height / patch_dim) % 2 == 0
        3) (patch_dim % 2) == 0
    """
    device=images.device

    b,c,h,w = images.size() # (b,c,h,w)
    row, col = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    col=col.to(device, non_blocking=True)
    row=row.to(device, non_blocking=True)

    col = col - (patch_dim * ( (row+col)%2 ) * ( 2*((col//patch_dim)%2)-1 ))

    row_flat = row.flatten()
    col_flat = col.flatten()
    idx : torch.Tensor = row_flat*w + col_flat

    images_flat = images.view(b, c, -1)
    idx = idx[None,None].expand(b, c, -1)

    images_swapped = torch.gather(images_flat, 2, idx).view(b, c, h, w)   

    return images_swapped
        
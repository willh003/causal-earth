

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

    b,_,h,w = images.size() # (b,c,h,w)
    row, col = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    col = col - (patch_dim * ( (row+col)%2 ) * ( 2*((col//patch_dim)%2)-1 ))
    images = images[torch.arange(b)[None], row[None], col[None]]

    return images
        
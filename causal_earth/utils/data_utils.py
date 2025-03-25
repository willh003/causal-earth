import numpy as np
import torch

def visualize_masked_image(image: torch.Tensor, mask: torch.Tensor, patch_size: int) -> np.ndarray:
    orig_img = image.detach().cpu().permute(1, 2, 0).numpy()
    mask = mask.detach().cpu().bool().numpy()
    
    # Normalize images for better visualization
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())

    # Expand mask to match image channels
    mask = np.transpose(mask.repeat(3, axis=0), [1,2,0])
    
    # Create masked original image
    masked_orig = orig_img.copy()
    masked_orig[mask] = 0.0  # White for masked areas
    
    return masked_orig
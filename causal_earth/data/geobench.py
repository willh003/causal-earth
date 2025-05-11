import torch
import numpy as np

class GeobenchCollator:
    """Custom collator for geobench dataset that extracts RGB bands and labels."""
    
    def __init__(self, transform=None):
        self.transform = transform
    
    def __call__(self, batch):
        # Extract RGB bands and labels from samples
        images = []
        labels = []
        
        for sample in batch:
            # Get RGB bands using pack_to_3d
            rgb_data, _ = sample.pack_to_3d(
                band_names=['red', 'green', 'blue'],
                resample=True,  # Resample to match largest shape
                fill_value=0,   # Fill missing bands with 0
                resample_order=3  # Use cubic interpolation
            )
            
            # Convert to torch tensor and normalize to [0, 1]
            rgb_tensor = torch.from_numpy(rgb_data).float()
            if rgb_tensor.max() > 1:
                rgb_tensor = rgb_tensor / 255.0
            
            rgb_tensor = rgb_tensor.permute(2, 0, 1)
            # Apply transforms if specified
            if self.transform:
                rgb_tensor = self.transform(rgb_tensor)
            
            images.append(rgb_tensor)
            
            # Handle both segmentation and classification labels
            if hasattr(sample.label, 'data'):  # If label is a Band object (segmentation)
                label_data = sample.label.data
                label_tensor = torch.from_numpy(label_data).long()
                labels.append(label_tensor)
            else:  # If label is a classification value
                labels.append(sample.label)
            
            
        
        # Stack images and labels into batches
        images = torch.stack(images)
        if isinstance(labels[0], torch.Tensor) and labels[0].ndim > 0:  # For segmentation
            labels = torch.stack(labels)
        else:  # For classification
            labels = torch.tensor(labels)
        
        return images, labels

class IJEPACollator:
    """Custom collator for IJEPA that generates block masks for target and context views.
    
    Generates:
    - 4 target block masks with random scale (0.15-0.2) and aspect ratio (0.75-1.5)
    - 1 context block mask with random scale (0.85-1.0) and unit aspect ratio
    - Ensures no overlap between context and target masks
    - Ensures consistent mask sizes across batch for efficient processing
    """
    
    def __init__(self, img_size=224, patch_size=16):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.h = self.w = int(self.num_patches ** 0.5)
        
    def _sample_block_mask(self, scale_range, aspect_range, num_blocks=1):
        """Sample block masks with given scale and aspect ratio ranges."""
        scales = torch.rand(num_blocks) * (scale_range[1] - scale_range[0]) + scale_range[0]
        aspects = torch.rand(num_blocks) * (aspect_range[1] - aspect_range[0]) + aspect_range[0]
        
        # Calculate block sizes
        block_areas = (scales * self.num_patches).long()
        block_heights = torch.sqrt(block_areas / aspects).long()
        block_widths = (block_areas / block_heights).long()
        
        # Ensure minimum size of 1
        block_heights = torch.clamp(block_heights, min=1)
        block_widths = torch.clamp(block_widths, min=1)
        
        # Sample random positions for blocks
        max_h = self.h - block_heights
        max_w = self.w - block_widths
        pos_h = torch.randint(0, max_h + 1, (num_blocks,))
        pos_w = torch.randint(0, max_w + 1, (num_blocks,))
        
        # Create masks
        masks = []
        for i in range(num_blocks):
            mask = torch.zeros(self.h, self.w)
            h_start, h_end = pos_h[i], pos_h[i] + block_heights[i]
            w_start, w_end = pos_w[i], pos_w[i] + block_widths[i]
            mask[h_start:h_end, w_start:w_end] = 1
            masks.append(mask)
            
        return torch.stack(masks)
    
    def __call__(self, batch):
        # Stack images into batch
        images = torch.stack([item[0] for item in batch])
        
        # Sample target masks (4 blocks)
        target_masks = self._sample_block_mask(
            scale_range=(0.15, 0.2),
            aspect_range=(0.75, 1.5),
            num_blocks=4
        )
        
        # Sample context mask (1 block)
        context_masks = self._sample_block_mask(
            scale_range=(0.85, 1.0),
            aspect_range=(1.0, 1.0),
            num_blocks=1
        )
        
        # Ensure consistent mask sizes across batch
        target_masks = target_masks.expand(len(batch), -1, -1, -1)
        context_masks = context_masks.expand(len(batch), -1, -1, -1)
        
        # Remove overlapping regions from context mask
        for i in range(len(batch)):
            context_mask = context_masks[i, 0]
            for j in range(4):
                target_mask = target_masks[i, j]
                context_mask = context_mask * (1 - target_mask)
            context_masks[i, 0] = context_mask
        
        # Reshape masks to patch sequence format
        target_masks = target_masks.reshape(len(batch), 4, -1)
        context_masks = context_masks.reshape(len(batch), 1, -1)
        
        return images, target_masks, context_masks

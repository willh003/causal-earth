import torch
import torch.nn.functional as F

class EarthNetCollator:
    """Custom collator for EarthNet2021xDataset that extracts and transforms RGB images.
    
    This collator:
    1. Extracts RGB bands from Sentinel-2 data
    2. Converts BGR to RGB
    3. Applies normalization transforms
    4. Returns batched images ready for model input
    """
    
    def __init__(self, transform=None):
        """Initialize the collator.
        
        Args:
            transform: Optional transform to apply to the images (e.g., normalization)
        """
        self.transform = transform
    
    def __call__(self, batch):
        """Process a batch of EarthNet data.
        
        Args:
            batch: List of dictionaries containing EarthNet data items
            
        Returns:
            torch.Tensor: Batched and transformed RGB images
        """
        # Extract RGB images from each sample
        images = []
        for data in batch:
            # Get Sentinel-2 data and extract RGB bands
            # [0] gets Sentinel-2 data (not E-OBS)
            # [0] gets first timestep
            # [1:4] gets BGR bands
            
            
            dynamic_bgr = data["dynamic"][0][0, 1:4, ...]
            
            # Convert BGR to RGB by flipping channels
            dynamic_rgb = dynamic_bgr.flip(0)

            images.append(dynamic_rgb)
        
        # Stack images into batch
        images = torch.stack(images)
        
        # Apply transforms if specified
        if self.transform:
            images = self.transform(images)

        return images 

class IJEPACollator:
    """Custom collator for I-JEPA training that generates context and target masks.
    
    This collator:
    1. Extracts RGB images from EarthNet data
    2. Generates 4 target block masks with random scale (0.15-0.2) and aspect ratio (0.75-1.5)
    3. Generates 1 context block mask with random scale (0.85-1.0) and unit aspect ratio
    4. Eliminates overlapping regions between context and target masks
    5. Returns batched images and masks ready for I-JEPA training
    """
    
    def __init__(self, transform=None, num_target_blocks=4, patch_size=16):
        """Initialize the collator.
        
        Args:
            transform: Optional transform to apply to the images
            num_target_blocks: Number of target blocks to generate (default: 4)
            patch_size: Size of each patch in pixels (default: 16)
        """
        self.transform = transform
        self.num_target_blocks = num_target_blocks
        self.patch_size = patch_size
        
    def _generate_block_mask(self, batch_size, scale_range, aspect_ratio_range, num_blocks=1):
        """Generate random block masks.
        
        Args:
            batch_size: Number of images in batch
            scale_range: Tuple of (min_scale, max_scale)
            aspect_ratio_range: Tuple of (min_ratio, max_ratio)
            num_blocks: Number of blocks to generate per image
            
        Returns:
            torch.Tensor: Binary masks of shape (batch_size, num_blocks, H, W)
        """
        min_scale, max_scale = scale_range
        min_ratio, max_ratio = aspect_ratio_range
        
        # Calculate grid dimensions
        grid_h = 224 // self.patch_size  # 14 for patch_size=16
        grid_w = 224 // self.patch_size
        
        # Generate random scales and aspect ratios
        scales = torch.rand(batch_size, num_blocks) * (max_scale - min_scale) + min_scale
        ratios = torch.rand(batch_size, num_blocks) * (max_ratio - min_ratio) + min_ratio
        
        # Calculate block dimensions
        block_areas = scales * grid_h * grid_w
        block_heights = torch.sqrt(block_areas / ratios)
        block_widths = block_areas / block_heights
        
        # Round to nearest integer
        block_heights = torch.round(block_heights).long()
        block_widths = torch.round(block_widths).long()
        
        # Ensure dimensions are within bounds
        block_heights = torch.clamp(block_heights, 1, grid_h)
        block_widths = torch.clamp(block_widths, 1, grid_w)
        
        # Generate random positions
        pos_h = torch.randint(0, grid_h - block_heights + 1, (batch_size, num_blocks))
        pos_w = torch.randint(0, grid_w - block_widths + 1, (batch_size, num_blocks))
        
        # Create masks
        masks = torch.zeros(batch_size, num_blocks, grid_h, grid_w)
        for b in range(batch_size):
            for n in range(num_blocks):
                h, w = block_heights[b, n], block_widths[b, n]
                y, x = pos_h[b, n], pos_w[b, n]
                masks[b, n, y:y+h, x:x+w] = 1
                
        return masks
    
    def __call__(self, batch):
        """Process a batch of EarthNet data.
        
        Args:
            batch: List of dictionaries containing EarthNet data items
            
        Returns:
            tuple: (images, context_masks, target_masks)
                - images: Batched and transformed RGB images
                - context_masks: Binary masks for context blocks
                - target_masks: Binary masks for target blocks
        """
        # Extract RGB images from each sample
        images = []
        for data in batch:
            # Get Sentinel-2 data and extract RGB bands
            dynamic_bgr = data["dynamic"][0][0, 1:4, ...]
            
            # Convert BGR to RGB by flipping channels
            dynamic_rgb = dynamic_bgr.flip(0)
            images.append(dynamic_rgb)
        
        # Stack images into batch
        images = torch.stack(images)
        batch_size = len(images)
        
        # Apply transforms if specified
        if self.transform:
            images = self.transform(images)
            
        # Generate target block masks
        target_masks = self._generate_block_mask(
            batch_size=batch_size,
            scale_range=(0.15, 0.2),
            aspect_ratio_range=(0.75, 1.5),
            num_blocks=self.num_target_blocks
        )
        
        # Generate context block mask
        context_masks = self._generate_block_mask(
            batch_size=batch_size,
            scale_range=(0.85, 1.0),
            aspect_ratio_range=(1.0, 1.0),
            num_blocks=1
        )
        
        # Eliminate overlapping regions
        for b in range(batch_size):
            # Get all target blocks for this image
            target_blocks = target_masks[b].sum(dim=0) > 0
            # Remove target regions from context mask
            context_masks[b, 0] = context_masks[b, 0] * (1 - target_blocks)
        
        return images, context_masks, target_masks 
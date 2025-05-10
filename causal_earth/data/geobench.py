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

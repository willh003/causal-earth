import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
import random
import os
from tqdm import tqdm

class MinicubesDataset(Dataset):
    """
    PyTorch Dataset for Zarr Minicubes store containing satellite imagery.
    
    Args:
        zarr_path (str): Path to the Zarr store
        transform (callable, optional): Optional transform to be applied to samples
        valid_only (bool): Whether to only return valid samples
        prefilter (bool): Whether to prefilter and cache valid indices
        cache_dir (str, optional): Directory to cache valid indices
        max_samples (int, optional): Maximum number of samples to use (for debugging)
    """
    def __init__(self, zarr_path, transform=None, valid_only=True, 
                 prefilter=True, cache_dir=None, max_samples=None):
        self.zarr_path = zarr_path
        self.transform = transform
        self.valid_only = valid_only
        self.cache_dir = cache_dir
        
        # Open Zarr store
        self.store = zarr.open(zarr_path)
        self.data_array = self.store['Minicubes']
        
        # Get dimensions
        self.time_dim, self.loc_dim = self.data_array.shape[0], self.data_array.shape[1]
        self.total_samples = self.time_dim * self.loc_dim
        
        # Get variables if available
        try:
            self.variables = self.store['variable'][:]
            self.red_idx = np.where(self.variables == 'B04')[0][0]
            self.green_idx = np.where(self.variables == 'B03')[0][0]
            self.blue_idx = np.where(self.variables == 'B02')[0][0]
        except (KeyError, IndexError):
            # Default to indices from your script
            self.red_idx = 2    # B04
            self.green_idx = 1  # B03
            self.blue_idx = 0   # B02
        
        # Cache of valid indices
        self.valid_indices = None
        
        # Try to load prefiltered indices from cache
        cache_loaded = False
        if prefilter and cache_dir:
            cache_file = os.path.join(cache_dir, 'valid_indices.npy')
            if os.path.exists(cache_file):
                try:
                    self.valid_indices = np.load(cache_file)
                    print(f"Loaded {len(self.valid_indices)} valid indices from cache")
                    cache_loaded = True
                except Exception as e:
                    print(f"Error loading cache: {e}")
        
        # Prefilter valid samples if requested
        if prefilter and not cache_loaded:
            self._prefilter_valid_samples()
            
            # Save cache if directory provided
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, 'valid_indices.npy')
                np.save(cache_file, self.valid_indices)
                print(f"Saved {len(self.valid_indices)} valid indices to cache")
        
        # Limit number of samples if requested
        if max_samples is not None and self.valid_indices is not None:
            self.valid_indices = self.valid_indices[:min(max_samples, len(self.valid_indices))]
            print(f"Limited to {len(self.valid_indices)} samples")
    
    def _prefilter_valid_samples(self):
        """Prefilter to find valid samples and cache their indices."""
        print("Prefiltering valid samples (this may take a while)...")
        valid_indices = []
        
        # Using tqdm for progress bar
        total = self.time_dim * self.loc_dim
        batch_size = 100  # Process in batches for better progress tracking
        
        for batch_start in tqdm(range(0, total, batch_size), desc="Filtering samples"):
            batch_indices = []
            for i in range(batch_start, min(batch_start + batch_size, total)):
                time_idx = i // self.loc_dim
                loc_idx = i % self.loc_dim
                batch_indices.append((time_idx, loc_idx))
            
            # Process batch by batch to reduce I/O overhead
            for time_idx, loc_idx in batch_indices:
                # Check if sample is valid
                try:
                    red = self.data_array[time_idx, loc_idx, self.red_idx, :, :]
                    green = self.data_array[time_idx, loc_idx, self.green_idx, :, :]
                    blue = self.data_array[time_idx, loc_idx, self.blue_idx, :, :]
                    
                    rgb_sample = np.stack([red, green, blue], axis=-1)
                    
                    if self._is_valid_sample(rgb_sample):
                        valid_indices.append(i)
                except Exception:
                    # Skip any problematic samples
                    continue
        
        self.valid_indices = np.array(valid_indices)
        print(f"Found {len(self.valid_indices)} valid samples out of {total}")
    
    def _is_valid_sample(self, sample):
        """Check if a sample is valid."""
        # Using the same validation as in your script
        valid = (0 < np.max(sample) < 10000) and np.min(sample) >= 0
        return valid
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.valid_indices is not None:
            return len(self.valid_indices)
        return self.time_dim * self.loc_dim
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Map index to time and location indices
        if self.valid_indices is not None:
            # If we have prefiltered indices, use them
            mapped_idx = self.valid_indices[idx]
            time_idx = mapped_idx // self.loc_dim
            loc_idx = mapped_idx % self.loc_dim
        else:
            # Otherwise use raw indices
            time_idx = idx // self.loc_dim
            loc_idx = idx % self.loc_dim
        
        # Get RGB channels
        red = self.data_array[time_idx, loc_idx, self.red_idx, :, :]
        green = self.data_array[time_idx, loc_idx, self.green_idx, :, :]
        blue = self.data_array[time_idx, loc_idx, self.blue_idx, :, :]
        
        # Stack to create RGB image
        rgb_sample = np.stack([red, green, blue], axis=-1)
        
        # If valid_only is True, ensure sample is valid (when not prefiltered)
        if self.valid_only and self.valid_indices is None:
            while not self._is_valid_sample(rgb_sample):
                # Try a new random sample
                time_idx = random.randint(0, self.time_dim - 1)
                loc_idx = random.randint(0, self.loc_dim - 1)
                
                red = self.data_array[time_idx, loc_idx, self.red_idx, :, :]
                green = self.data_array[time_idx, loc_idx, self.green_idx, :, :]
                blue = self.data_array[time_idx, loc_idx, self.blue_idx, :, :]
                
                rgb_sample = np.stack([red, green, blue], axis=-1)
        
        # Scale the data as in your display function
        rgb_sample = rgb_sample / 10000.0
        
        # Convert to PyTorch tensor
        sample_tensor = torch.tensor(rgb_sample, dtype=torch.float32)
        
        # Apply any transformations
        if self.transform:
            sample_tensor = self.transform(sample_tensor)
        
        # Return sample and metadata
        return {
            'image': sample_tensor,
            'coords': (time_idx, loc_idx)
        }


def create_minicubes_dataloader(zarr_path, batch_size=16, num_workers=4, 
                               valid_only=True, prefilter=True, 
                               cache_dir='./cache', max_samples=None,
                               transform=None, shuffle=True):
    """
    Create a DataLoader for the Minicubes dataset.
    
    Args:
        zarr_path (str): Path to the Zarr store
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        valid_only (bool): Whether to only return valid samples
        prefilter (bool): Whether to prefilter and cache valid indices
        cache_dir (str): Directory to cache valid indices
        max_samples (int, optional): Maximum number of samples to use
        transform (callable, optional): Optional transform to be applied to samples
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    # Create dataset
    dataset = MinicubesDataset(
        zarr_path=zarr_path,
        transform=transform,
        valid_only=valid_only,
        prefilter=prefilter,
        cache_dir=cache_dir,
        max_samples=max_samples
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    import time
    from torchvision import transforms
    
    # Path to zarr store
    zarr_path = "../data"
    
    # Example transformations (optional)
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # Convert HWC to CHW format
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    # Create dataloader
    dataloader = create_minicubes_dataloader(
        zarr_path=zarr_path,
        batch_size=8,
        num_workers=2,
        valid_only=True,
        prefilter=True,
        cache_dir='./cache',
        max_samples=100,  # Limit samples for testing
        transform=transform
    )
    
    print(f"DataLoader created with {len(dataloader.dataset)} samples")
    
    # Iterate through a few batches as a test
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Just test a few batches
            break
            
        images = batch['image']
        coords = batch['coords']
        
        print(f"Batch {i}: {images.shape}, {images.min().item():.4f} to {images.max().item():.4f}")
        print(f"Coordinates: {coords}")
        
        # Optional: Visualize a sample
        if i == 0:
            import matplotlib.pyplot as plt
            
            # Take first image in batch and convert back to HWC format
            img = images[0].permute(1, 2, 0).cpu().numpy()
            
            # Denormalize if needed
            img = img * 0.25 + 0.5
            img = np.clip(img, 0, 1)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title("Sample from DataLoader")
            plt.axis('off')
            plt.savefig('sample_from_dataloader.png')
            plt.close()
    
    elapsed = time.time() - start_time
    print(f"Processed {min(3, len(dataloader))} batches in {elapsed:.2f} seconds")
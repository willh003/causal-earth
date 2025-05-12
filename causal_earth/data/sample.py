import xarray as xr
import matplotlib.pyplot as plt
import os
from causal_earth import extract_all_rgb
import torch
from torch.utils.data import Dataset, DataLoader


class PooledRGBDataset(Dataset):
    """Each Minicube contains several RGB images of the same location through time.
    This pools all of the images into one large dataset."""

    def __init__(self, directory):
        self.directory = directory
        self.file_paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".nc") or f.endswith(".zarr")
        ]

        self.image_locations = []
        for file_idx, file_path in enumerate(self.file_paths):
            with xr.open_dataset(file_path) as dataset:
                rgb_images = extract_all_rgb(dataset)
                for img_idx, _ in enumerate(rgb_images):
                    self.image_locations.append((file_idx, img_idx))

        print(
            f"Found {len(self.image_locations)} RGB images across {len(self.file_paths)} files"
        )

    def __len__(self):
        return len(self.image_locations)

    def __getitem__(self, idx):
        file_idx, img_idx = self.image_locations[idx]
        file_path = self.file_paths[file_idx]

        with xr.open_dataset(file_path) as dataset:
            rgb_images = extract_all_rgb(
                dataset
            )  # NOTE: A bit wasteful since we load ~27 images and only keep 1.
            return rgb_images[img_idx]


def create_pooled_rgb_dataset(directory, batch_size=32, num_workers=8, shuffle=True):
    dataset = PooledRGBDataset(directory)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader


def display_rgb_grid(batch, grid=(3, 5), save_path=None):
    """
    Display a grid (rows, cols) of RGB images from a batch with shape (4, 3, 128, 128)
    """
    rows, cols = grid
    _, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axs = axs.flatten()
    imgs = batch.permute(0, 2, 3, 1).cpu().numpy()
    for i in range(rows * cols):
        # print(F"max:{batch.max()}, min:{batch.min()}")
        # NOTE: some values are NaN
        img = imgs[i]  # Convert from (C,H,W) to (H,W,C)
        axs[i].imshow(img)
        axs[i].set_title(f"Sentinel-2 RGB Images {i+1}")
        axs[i].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    rows, cols = 3, 5
    rgb_loader = create_pooled_rgb_dataset(
        "C:/Users/avery/Desktop/AI Prac/greenearthnet/earthnet2021x/train/29SPC",
        num_workers=4,
        batch_size=rows * cols,
    )

    for i, batch in enumerate(rgb_loader):
        display_rgb_grid(batch, (rows, cols), save_path=f"examples/sample_{i}")
        if i == 3:
            print("Ending sampler preview.")
            break

import zarr
import numpy as np
import matplotlib.pyplot as plt
import random

zarr_path = "../data"

def is_valid_sample(sample):
    "Check if a sample is valid."
    # NOTE: These conditions are heuristic
    valid = (0 < np.max(sample) < 10000) and np.min(sample) >= 0
    print(F"({0 < np.max(sample)} && {np.max(sample)< 10000}) && {np.min(sample) >= 0}")
    return valid

def get_random_valid_sample(africa_zarr, verbose=False):
    "Get a random valid RGB sample Africa-Minicubes .zarr store."
    data_array = africa_zarr['Minicubes']
    
    # Get dimensions from the data array
    time_dim, loc_dim = data_array.shape[0], data_array.shape[1]
    print(f"Array dimensions: time={time_dim}, locations={loc_dim}")
    
    while True:
        time_idx = random.randint(0, time_dim - 1)
        loc_idx = random.randint(0, loc_dim - 1)
        
        red = data_array[time_idx, loc_idx, 2, :, :]
        green = data_array[time_idx, loc_idx, 1, :, :]
        blue = data_array[time_idx, loc_idx, 0, :, :]
        
        rgb_sample = np.stack([red, green, blue], axis=-1) # image: (128, 128, 3=RGB)
        
        if is_valid_sample(rgb_sample):
            time_var = africa_zarr['time'][:]
            time_label = time_var[time_idx] if time_idx < len(time_var) else f"time_{time_idx}"
            print(f"Sample from time: {time_label}, location: {loc_idx}")
            
            return rgb_sample / 10000, (time_idx, loc_idx)
        elif verbose:
            print("Invalid sample, resampling...")
        
def display_raw_image(sample):
    "Display a raw RGB image from the sample"
    sample_scaled = sample
    plt.figure(figsize=(10, 10))
    plt.imshow(sample_scaled)
    plt.title("Raw Sentinel-2 RGB Image (R=B04, G=B03, B=B02)")
    plt.axis('off')
    plt.show()

# Main execution
if __name__ == "__main__":
    store = zarr.open(zarr_path) 
    sample, indices = get_random_valid_sample(store, verbose=True)
    display_raw_image(sample)
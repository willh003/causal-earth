import zarr
import numpy as np
import xarray as xr

# NOTE: I use a symbolic link to `africa_minicubes.zarr` at ../data 
dataset_path = "../data" 

zarr_store = zarr.open_consolidated(dataset_path)
# Check the actual values in the time array
print("Time values:")
print(zarr_store['time'][:])

# Sample a few minicubes to see if they have consistent data patterns
for i in range(2589): 
    # Count non-NaN values across the time dimension for the first variable at position (0,0)
    valid_count = np.sum(~np.isnan(zarr_store['Minicubes'][i, :, 0, 0, 0]))
    print(f"Minicube {i} has {valid_count} valid time points out of 160")


# TODO: Get xarray working
# ds = xr.open_zarr(dataset_path)
# print(ds)


#----------------------------------#
# Inspect the Zarr store structure #
#----------------------------------#
"""
zarr_store = zarr.open_consolidated(dataset_path)
print("Zarr store structure:")
print(zarr_store.tree())

Get basic information about the arrays
for key in zarr_store.keys():
    if isinstance(zarr_store[key], zarr.core.Array):
        print(f"\nArray: {key}")
        print(f"  Shape: {zarr_store[key].shape}")
        print(f"  Dtype: {zarr_store[key].dtype}")
        print(f"  Chunks: {zarr_store[key].chunks}")
"""
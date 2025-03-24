import xarray as xr
import os
import numpy as np


filepath = "/home/wph52/greenearthnet/train/34VFQ/34VFQ_2018-05-02_2018-09-28_4921_5049_5305_5433_76_156_82_162.npz"

print(os.path.exists(filepath))
#y = xr.open_dataset(filepath)
x = np.load(filepath)
print(x.files)

breakpoint()
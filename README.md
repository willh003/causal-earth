### Setup

The repo is set up to contain the earthnet-models-pytorch submodule, which implements data loading from earthnet. You should install it as editable so you can make changes as needed to the dataloaders. 

```
git clone --recurse-submodules git@github.com:willh003/CausalEarth.git

conda create -n earth python=3.10
conda activate earth
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge numpy matplotlib pillow xarray zarr netcdf4 scipy imageio networkx scikit-image s3fs pyproj pyarrow seaborn cartopy tensorboard dask pytorch-lightning=1.7.7 torchmetrics=0.7.3
pip install earthnet segmentation-models-pytorch wandb draccus

pip install -e ./earthnet-models-pytorch
pip install -e .
```



### Dev
- Download scripts are in `causal-earth/download` 
- Model interfaces are in `causal-earth/models` 
- Train/eval scripts are in `causal-earth/scripts`
- Utils are in `causal-earth/utils`
- Torch Dataset implementations are in  `earthnet-models-pytorch/earthnet_models_pytorch/data`
- Configs are defined as dataclasses, and loaded into scripts using draccus


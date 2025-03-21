### Setup

The repo is set up to contain the earthnet-models-pytorch submodule, which implements data loading from earthnet. You should install it as editable so you can make changes as needed to the dataloaders. 

```
git clone --recurse-submodules git@github.com:willh003/CausalEarth.git

conda create -n earth python=3.10
conda activate earth
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt

pip install -e ./earthnet-models-pytorch
pip install -e .
```

### Training
For now, we are using SatMAE pretrained checkpoints.

train_mae.py uses the checkpoint fmow_pretrain.pth from 
https://zenodo.org/records/7369797

Download the checkpoint:
```
curl -L "https://zenodo.org/records/7369797/files/fmow_pretrain.pth?download=1" -o "causal_earth/ckpts/fmow_pretrain.pth" --progress-bar
```

### Dev
- Download scripts are in `causal-earth/download` 
- Model interfaces are in `causal-earth/models` 
- Train/eval scripts are in `causal-earth/scripts`
- Utils are in `causal-earth/utils`
- Torch Dataset implementations are in  `earthnet-models-pytorch/earthnet_models_pytorch/data`
- Configs are defined as dataclasses, and loaded into scripts using draccus


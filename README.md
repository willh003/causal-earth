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

### Data
Data collection scripts are in causal_earth/scripts. Note that with 10,000 samples (current setup), the dataset is ~30G. If converted to fast access, this becomes ~120G.
1. Download using `download.sh`. You can specify the save dir and number of samples
2. Create validation split using `create_val_split.sh`. You can specify the val percent and the base dir (most likely save dir from the previous step + "earthnet2021x")
3. Create fast access data (convert from .nc to .npz) for train and val splits using `create_fast_access.sh`. This speeds up training by 2-3x, at the cost of 4x disk space for the dataset. [NOTE]: this step is not necessary. If you do not do it, make sure you set FAST_ACCESS=false in the train script.

### Training
For now, we are using SatMAE pretrained checkpoints.

train_mae.py uses the checkpoint fmow_pretrain.pth from 
https://zenodo.org/records/7369797

1. Download the checkpoint:
    ```
    curl -L "https://zenodo.org/records/7369797/files/fmow_pretrain.pth?download=1" -o "causal_earth/ckpts/fmow_pretrain.pth" --progress-bar
    ```

2. Set up a wandb API key (see wandb docs)

3. In scripts/, modify `train.sh` to your needs. Be sure to specify TRAIN_DIR, VAL_DIR, FAST_ACCESS, and CKPT_PATH.
4. Run the training script from scripts/:
    ```
    cd scripts
    source train.sh
    ```

### Dev
- Download scripts are in `causal-earth/download` 
- Model interfaces are in `causal-earth/models` 
- Train/eval scripts are in `causal-earth/scripts`
- Utils are in `causal-earth/utils`
- Torch Dataset implementations are in  `earthnet-models-pytorch/earthnet_models_pytorch/data`
- Configs are defined as dataclasses, and loaded into scripts using draccus


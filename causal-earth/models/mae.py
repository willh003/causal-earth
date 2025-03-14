

import torch
import torch.nn as nn

class TemporalMAE(nn.Module):
    """
    A masked autoencoder that reconstructs an image in the future
    (may be able to use standard MAE with a different loss)
    https://arxiv.org/abs/2111.06377
    """

    def __init__(self):
        __super__().init()

    def forward(x):
        """
        x is a set of latent embeddings representing a (possibly multimodal) input
        """
        pass

class MAE(nn.Module):
    """
    A Masked Autoencoder, from He, et al.
    https://arxiv.org/abs/2111.06377
    """

    def __init__(self):
        __super__().init()

    def forward(x):
        """
        x is a set of latent embeddings representing a (possibly multimodal) input
        """
        pass
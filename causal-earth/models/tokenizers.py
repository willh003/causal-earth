


class JointImageWeatherTokenizer(nn.Module):
    """
    Project an image at time t and weather at time t+1 into a shared embedding space 
    """

    def __init__(self):
        __super__().init()

    def forward(image, weather):
        """
        x is a set of latent embeddings representing a (possibly multimodal) input
        """
        pass

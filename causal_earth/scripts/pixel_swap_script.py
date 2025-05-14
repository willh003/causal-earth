from PIL import Image
import numpy as np
import argparse
from utils.train_utils import pixel_swap

if __name__=="__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('image_path')

    args = argparser.parse_args()
    Image.fromarray(pixel_swap(np.asarray(Image.open(args.image_path)).copy(),patch_size=16)).save(F"swap_{args.image_path}")

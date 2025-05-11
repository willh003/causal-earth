from causal_earth.data.sample import create_pooled_rgb_dataset
from argparse import ArgumentParser
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import io
import streamlit as st
from earthnet_models_pytorch.data.en21x_data import EarthNet2021XDataset
from causal_earth.models import mae_vit_large_patch16_dec512d8b
from causal_earth.cfgs import MAEConfig
import draccus
import torch
from causal_earth.utils import interpolate_pos_embed
import time
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from causal_earth.utils.model_utils import get_attn_maps
from PIL import Image


def generate_image(batch, grid=(1, 1)):
    """
    Returns a grid (rows, cols) of RGB images from a batch with shape (4, 3, 128, 128)
    """
    rows, cols = grid
    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    axs = axs.flatten()
    if grid[0] == 1 and grid[1] == 1:
        batch = batch.unsqueeze(0)
    imgs = batch.permute(0, 2, 3, 1).cpu().numpy()
    for i in range(rows * cols):
        # print(F"max:{batch.max()}, min:{batch.min()}")
        # NOTE: some values are NaN
        img = imgs[i]  # Convert from (C,H,W) to (H,W,C)
        axs[i].imshow(img)
        axs[i].set_title(f"Sentinel-2 RGB Images {i+1}")
        axs[i].axis("off")

    fig.tight_layout()
    return fig


def streamlit_app():
    # Initialize Streamlit session state with default values if not already set.
    if "current_index" not in st.session_state:
        # Set default index for the session state
        st.session_state.current_index = 0
    if "filters" not in st.session_state:
        # Set default filters for the session state
        st.session_state.filters = {
            "include_clouds": False,
            "attention_mask": False,
        }

    if "train_loader" not in st.session_state:
        st.session_state.train_loader = DataLoader(
            st.session_state.train_set,
            batch_size=1,  # for now set to 1 so that all images are loaded
            shuffle=True,
            num_workers=1,  # please change I have a shitty computer that crashes if more than 4
            pin_memory=torch.cuda.is_available(),
        )
        print("train_loader created")

    def generate_attention_mask(image):
        """
        Generates an attention mask for the given image.
        The mask is a binary tensor with the same shape as the image, where pixels with values > 0 are set to 1,
        and others are set to 0.
        """
        # Assuming the image is in the format (C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        # Evaluate the image in model
        model = st.session_state.model
        image = st.session_state.transform(image)  # normalize the rgb image
        loss, pred_images, mask = model(
            image, mask_ratio=st.session_state.app_cfg.mask_ratio
        )
        # Extract attention weights from the model output
        # attentions = output[1]  # Assuming the second element contains attention weights
        # print(model.modules())
        return get_attn_maps(model, image, mask)
        # rgb *= attention
        # Get the attention mask from the model output

    def current_image():
        """Returns the current image as a BytesIO object for Streamlit."""
        # Get the current image from the DataLoader
        curr_data_sen2 = st.session_state.train_loader.dataset[
            st.session_state.current_index
        ]["dynamic"][
            0
        ]  # only get first image for now
        red = curr_data_sen2[0, 3, :, :].numpy()
        green = curr_data_sen2[0, 2, :, :].numpy()
        blue = curr_data_sen2[0, 1, :, :].numpy()
        rgb = torch.from_numpy(np.stack([red, green, blue], axis=0))
        if not st.session_state.filters["include_clouds"]:
            # Remove clouds from the image if the filter is set
            cloud_mask = (
                st.session_state.train_loader.dataset[st.session_state.current_index][
                    "dynamic_mask"
                ][0]
                < 1.0
            )
            rgb = (
                rgb * cloud_mask[0][0]
            )  # Only take the cloud mask from the first image

        if st.session_state.filters["attention_mask"]:
            # Apply attention mask if the filter is set
            attention_mask = generate_attention_mask(rgb)
            rgb = Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            rgb.paste(im=attention_mask, box=(0, 0), mask=attention_mask)
            # attention_mask.permute(2, 0, 1)
            # print(attention_mask)
            fig, axs = plt.subplots(1, 1)
            axs.imshow(rgb)
            axs.axis("off")
        # Convert the image to a grid format
        else:
            fig = generate_image(rgb, grid=(1, 1))
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        plt.close(fig)
        img_buf.seek(0)
        return img_buf

    st.title("Causal Earth Data Applet")

    previous_image_button_col, curr_image_col, next_image_button_col = st.columns(
        [1, 2, 1]
    )

    def on_next_image():
        st.session_state.current_index += 1
        if st.session_state.current_index >= len(st.session_state.train_loader):
            st.session_state.current_index = 0

    def on_prev_image():
        st.session_state.current_index -= 1
        if st.session_state.current_index < 0:
            st.session_state.current_index = len(st.session_state.train_loader) - 1

    st.sidebar.title("Filters")

    def toggle_include_clouds():
        st.session_state.filters["include_clouds"] = not st.session_state.filters[
            "include_clouds"
        ]
        # Update the DataLoader based on the filter

    st.sidebar.checkbox(
        "Include Clouds",
        value=st.session_state.filters["include_clouds"],
        on_change=toggle_include_clouds,
        key="include_clouds_checkbox",
    )

    def toggle_attention_mask():
        st.session_state.filters["attention_mask"] = not st.session_state.filters[
            "attention_mask"
        ]

    st.sidebar.checkbox(
        "Attention Mask",
        value=st.session_state.filters["attention_mask"],
        on_change=toggle_attention_mask,
        key="attention_mask_checkbox",
    )

    with next_image_button_col:
        st.button("Next Image", on_click=on_next_image, key="next_image")
    with previous_image_button_col:
        st.button("Previous Image", on_click=on_prev_image, key="prev_image")

    with curr_image_col:
        st.image(
            current_image(),
            caption=f"Current Image: {st.session_state.current_index}",
            use_container_width=True,
            channels="RGB",
            output_format="auto",
        )

    st.write(f"Total Images: {len(st.session_state.train_loader)}")


def load_pretrained_weights(model, ckpt_path):
    """Load pretrained weights into the model."""
    print(f"Loading pre-trained checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()

    # Handle incompatible keys
    incompatible_keys = [
        "pos_embed",
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "head.weight",
        "head.bias",
    ]
    for k in incompatible_keys:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Handle positional embeddings
    interpolate_pos_embed(model, checkpoint_model)

    # Load state dict
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    return model


@draccus.wrap()
def main(cfg: MAEConfig):
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        cfg.train_dir = (
            "../greenearthnet/earthnet2021x/train/29SPC"  # hardcoded for now
        )
        if "train_set" not in st.session_state:
            st.session_state.train_set = None
        if "model" not in st.session_state:
            st.session_state.model = None
        if "app_cfg" not in st.session_state:
            st.session_state.app_cfg = cfg
        if "transform" not in st.session_state:
            st.session_state.transform = transforms.Compose(
                [
                    # Resize the image to 224x224 pixels
                    transforms.Lambda(
                        lambda x: F.interpolate(
                            x, size=(224, 224), mode="bilinear", align_corners=False
                        )
                    ),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        # global train_set, model, app_cfg

        st.session_state.train_set = EarthNet2021XDataset(
            cfg.train_dir, dl_cloudmask=True, allow_fastaccess=cfg.allow_fastaccess
        )
        val_set = (
            EarthNet2021XDataset(
                cfg.val_dir, dl_cloudmask=True, allow_fastaccess=cfg.allow_fastaccess
            )
            if cfg.val_dir
            else None
        )

        train_loader = DataLoader(
            st.session_state.train_set,
            batch_size=1,  # for now set to 1 so that all images are loaded
            shuffle=True,
            num_workers=1,  # please change I have a shitty computer that crashes if more than 4
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = (
            DataLoader(
                val_set,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            if val_set
            else None
        )

        st.session_state.model = mae_vit_large_patch16_dec512d8b()
        # Load pre-trained weights if provided
        if cfg.ckpt_path:
            st.session_state.model = load_pretrained_weights(
                st.session_state.model, cfg.ckpt_path
            )
        st.session_state.initialized = True

    streamlit_app()


if __name__ == "__main__":
    main()

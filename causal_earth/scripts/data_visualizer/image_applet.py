from causal_earth.data.sample import create_pooled_rgb_dataset
from argparse import ArgumentParser
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import io
import streamlit as st
from earthnet_models_pytorch.data.en21x_data import EarthNet2021XDataset
from causal_earth.cfgs import MAEConfig
import draccus
import torch
import time


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


def streamlit_app(train_set: Dataset):

    # Initialize Streamlit session state with default values if not already set.
    st.session_state.setdefault("current_index", 0)
    st.session_state.setdefault("filters", {"include_clouds": False})
    st.session_state.setdefault(
        "train_loader",
        DataLoader(
            train_set,
            batch_size=1,  # for now set to 1 so that all images are loaded
            shuffle=True,
            num_workers=1,  # please change I have a shitty computer that crashes if more than 4
            pin_memory=torch.cuda.is_available(),
        ),
    )

    def current_image():
        """Returns the current image as a BytesIO object for Streamlit."""
        # Get the current image from the DataLoader
        curr_data_sen2 = st.session_state.train_loader.dataset[
            st.session_state.current_index
        ]["dynamic"][
            0
        ]  # only get first image for now

        if not st.session_state.filters["include_clouds"]:
            # Remove clouds from the image if the filter is set
            cloud_mask = (
                st.session_state.train_loader.dataset[st.session_state.current_index][
                    "dynamic_mask"
                ][0]
                < 1.0
            )
            curr_data_sen2 = curr_data_sen2 * cloud_mask
        # Convert the image to a grid format
        red = curr_data_sen2[0, 3, :, :].numpy()
        green = curr_data_sen2[0, 2, :, :].numpy()
        blue = curr_data_sen2[0, 1, :, :].numpy()
        rgb = torch.from_numpy(np.stack([red, green, blue], axis=0))
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
    print(st.session_state.current_index)


@draccus.wrap()
def main(cfg: MAEConfig):
    cfg.train_dir = "../greenearthnet/earthnet2021x/train/29SPC"  # hardcoded for now
    print(cfg.train_dir)
    train_set = EarthNet2021XDataset(
        cfg.train_dir, dl_cloudmask=True, allow_fastaccess=cfg.allow_fastaccess
    )
    val_set = (
        EarthNet2021XDataset(
            cfg.val_dir, dl_cloudmask=True, allow_fastaccess=cfg.allow_fastaccess
        )
        if cfg.val_dir
        else None
    )

    print(f"Train set size: {len(train_set)}")
    time1 = time.time()
    train_loader = DataLoader(
        train_set,
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
    print(f"Train loader size: {len(train_loader)}")
    # print(len(rgbloader))
    streamlit_app(train_set)


if __name__ == "__main__":
    main()

import tkinter as tk
from causal_earth.data.sample import create_pooled_rgb_dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
import io
import streamlit as st


if "current_index" not in st.session_state:
    st.session_state.current_index = 0


def generate_image(batch, grid=(1, 1)):
    """
    Returns a grid (rows, cols) of RGB images from a batch with shape (4, 3, 128, 128)
    """
    rows, cols = grid
    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    axs = axs.flatten()
    print(batch.shape)
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


def streamlit_app(rgbloader: DataLoader):
    def current_image():
        fig = generate_image(
            rgbloader.dataset[st.session_state.current_index], grid=(1, 1)
        )
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
        if st.session_state.current_index >= len(rgbloader):
            st.session_state.current_index = 0

    def on_prev_image():
        st.session_state.current_index -= 1
        if st.session_state.current_index < 0:
            st.session_state.current_index = len(rgbloader) - 1

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

    st.write(f"Total Images: {len(rgbloader)}")
    print(st.session_state.current_index)


if __name__ == "__main__":
    parser = ArgumentParser(description="Process some integers.")
    parser.add_argument("-data_path", type=str)

    args = parser.parse_args()

    # print(args.data_path)
    rgbloader = create_pooled_rgb_dataset(
        directory=args.data_path,
        batch_size=1,
        num_workers=1,  # please change I have a shitty computer that crashes if more than 4
        shuffle=True,
    )
    # print(len(rgbloader))
    streamlit_app(rgbloader)

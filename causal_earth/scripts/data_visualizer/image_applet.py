import tkinter as tk
from causal_earth.data.sample import create_pooled_rgb_dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
import io


global current_index
current_index = 0


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


def main(rgbloader: DataLoader):
    def current_image():
        global current_index
        fig = generate_image(rgbloader.dataset[current_index], grid=(1, 1))
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        plt.close(fig)
        img = Image.open(img_buf)
        return ImageTk.PhotoImage(image=img)

    def on_next_image():
        global current_index
        current_index += 1
        if current_index >= len(rgbloader):
            current_index = 0
        label2.config(text="Current Image: " + str(current_index))
        image = current_image()
        image_panel.config(image=image)
        image_panel.image = image

    def on_prev_image():
        global current_index
        current_index -= 1
        if current_index < 0:
            current_index = len(rgbloader) - 1
        label2.config(text="Current Image: " + str(current_index))
        image = current_image()
        image_panel.config(image=image)
        image_panel.image = image

    root = tk.Tk()
    root.title("Casaul Earth Data Applet")
    next_image_button = tk.Button(
        text="next image", command=on_next_image, fg="darkgreen", bg="white"
    )
    next_image_button.pack()

    prev_image_button = tk.Button(
        text="previous image", command=on_prev_image, fg="darkgreen", bg="white"
    )
    prev_image_button.pack()

    image = current_image()
    image_panel = tk.Label(root, image=image)
    image_panel.image = image
    image_panel.pack()

    # image = FigureCanvasTkAgg(update_image(), master=root)
    # image.draw(),
    # image.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    label1 = tk.Label(root, text=str(len(rgbloader)) + " images")
    label1.pack(pady=20)

    label2 = tk.Label(root, text="Current Image: " + str(current_index))
    label2.pack(pady=20)

    root.mainloop()
    # hello world


if __name__ == "__main__":
    parser = ArgumentParser(description="Process some integers.")
    parser.add_argument("-data_path", type=str)

    args = parser.parse_args()

    print(args.data_path)
    rgbloader = create_pooled_rgb_dataset(
        directory=args.data_path,
        batch_size=1,
        num_workers=1,  # please change I have a shitty computer that crashes if more than 4
        shuffle=True,
    )
    print(len(rgbloader))
    main(rgbloader)

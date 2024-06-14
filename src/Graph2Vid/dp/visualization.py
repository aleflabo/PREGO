import io
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image


# defining the colors and shapes
color_code = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "grey",
    "olive",
    "cyan",
    "lime",
    "grey",
    "firebrick",
    "coral",
    "chocolate",
    "saddlebrown",
    "bisque",
    "goldenrod",
    "gold",
    "khaki",
    "darkolivegreen",
    "greenyellow",
    "palegreen",
    "springgreen",
    "aquamarine",
    "teal",
    "deepskyblue",
    "navy",
    "mediumslateblue",
    "royalblue",
    "indigo",
    "magenta",
    "deeppink",
    "crimson",
    "violet",
    "snow",
    "lightgrey",
    "wheat",
    "dodgerblue",
    "darkseagreen",
]
color_code = color_code * 10
shape_code = ["o", "s", "P", "*", "h", ">", "X", "d", "D", "v", "<", "p"]
shape_code = shape_code * int(len(color_code) / len(shape_code) + 1)

color_values = []
for color in color_code:
    _ = plt.fill([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], color)
    buf = io.BytesIO()
    _ = plt.savefig(buf, format="png")
    _ = plt.close()
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    color_values.append(img[100, 300])

color_code_hex = []
for color_value in color_values:
    step_color_rgb = tuple([s.item() for s in color_value])
    color_code_hex.append("#%02x%02x%02x" % step_color_rgb)


def plot_alignment(
    step_ids, frame_labels, step_colors, step_shapes, size=(15, 2), do_legend=True, name="all_step_to_video", to_np=True
):
    N_steps = len(frame_labels)

    plt.rcParams["figure.figsize"] = (size[0], size[1])
    ax = plt.subplot(1, 1, 1)
    _ = ax.set_title(name)

    tick_freq = 50 if N_steps > 1500 else 20
    _ = plt.xticks(np.arange(0, N_steps, tick_freq))
    _ = plt.xlim(0, N_steps)
    _ = plt.tick_params(bottom=True, top=False, left=True, right=True, labelright=True)
    _ = plt.grid()

    for si, step_id in enumerate(step_ids):
        time, val = [], []
        for i in range(N_steps):
            if si + 1 == frame_labels[i]:
                time.append(i)
                val.append(1)
        time, val = np.array(time), np.array(val)
        _ = plt.plot(time, val, step_shapes[step_id], color=step_colors[step_id], label=str(step_id))

    N = len(frame_labels)
    if do_legend:
        _ = plt.xlim(0, N + int(0.10 * N))
        plt.legend()

    if to_np:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGB"))
        return img
    else:
        return plt


def plot_step_to_video_alignment(corresp_mat, size=(15, 2)):
    """corresp_mat is of shape [K, N], where K is num_steps, and N is video_len"""
    step_ids = np.arange(corresp_mat.size(0)) + 1
    labels = corresp_mat.to(float).argmax(0) + 1 * corresp_mat.to(bool).any(0)

    K_present = corresp_mat.to(bool).any(1).to(int).sum().item()
    name = f"Video Segmentation | {K_present} steps present"
    return plot_alignment(step_ids, labels, color_code, shape_code, name=name, size=size)


def plot_similarities(
    sim, drop_line=None, colors=None, select=None, color_offset=0, do_legend=True, name="", size=(15, 2), to_np=True
):
    colors = colors if colors is not None else color_code
    K, N = sim.shape
    select = select if select is not None else np.arange(K)

    plt.rcParams["figure.figsize"] = (size[0], size[1])
    ax = plt.subplot(1, 1, 1)
    _ = ax.set_title(name)

    _ = plt.xticks(np.arange(0, N, 20))
    _ = plt.xlim(0, N)
    _ = plt.tick_params(bottom=True, top=False, left=True, right=True, labelright=True)
    _ = plt.grid()

    for i in range(K):
        if i in select:
            _ = plt.plot(np.arange(N), sim[i], color=colors[i + color_offset], label=str(i))

    if drop_line is not None:
        _ = plt.plot(np.arange(N), drop_line * np.ones(N), color="black")

    if do_legend:
        _ = plt.xlim(0, N + int(0.10 * N))
        plt.legend()

    if to_np:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGB"))
        return img
    else:
        return plt


def plot_gt_seg(N, starts, ends, colors, shapes, name="GT Seg", clip_len=1):
    K = len(starts)
    labels = -np.ones(N)
    for i in range(K):
        s, e = int(starts[i]), int(ends[i])
        labels[s : e + 1] = i + 1
    step_ids = np.arange(K)
    return plot_alignment(step_ids, labels, colors, shapes, to_np=False, name=name)

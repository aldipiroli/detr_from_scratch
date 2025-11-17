import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import detr.utils.misc as utils


def convert_vertex_to_xyhw(boxes):
    xmin = boxes[..., 0]
    ymin = boxes[..., 1]
    xmax = boxes[..., 2]
    ymax = boxes[..., 3]

    c_x = (xmin + xmax) / 2
    c_y = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return torch.stack([c_x, c_y, w, h], dim=-1)


def get_figure(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    img = img[:, :, :3]
    plt.close(fig)
    return img


def plot_boxes(boxes, ax, color="lime", linewidth=3):
    for box in boxes:
        x, y, w, h = box
        x0 = x - w / 2
        y0 = y - h / 2
        rect = patches.Rectangle((x0, y0), h, w, linewidth=linewidth, edgecolor=color, facecolor="none")
        ax.add_patch(rect)


def plot_img_with_boxes(image, gt_boxes=None, pred_boxes=None, output_path="tmp", alpha=1, return_figure=False):
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    image = utils.to_cpu(image)

    # plot img
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, alpha=alpha)

    # plot boxes
    if gt_boxes is not None:
        gt_boxes = utils.to_cpu(convert_vertex_to_xyhw(gt_boxes))
        plot_boxes(gt_boxes, ax, color="lime")

    if pred_boxes is not None:
        pred_boxes = utils.to_cpu(convert_vertex_to_xyhw(pred_boxes))
        plot_boxes(pred_boxes, ax, color="red")

    if return_figure:
        img = get_figure(fig)
        img = img.transpose(2, 0, 1)
        return img
    else:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

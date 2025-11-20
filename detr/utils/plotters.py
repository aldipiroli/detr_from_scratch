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


def plot_boxes(boxes, ax, pred_cls_idx=None, pred_cls_prob=None, color="lime", linewidth=5):
    if pred_cls_idx is None:
        pred_cls_idx = [None] * len(boxes)
    if pred_cls_prob is None:
        pred_cls_prob = [None] * len(boxes)

    for box, idx, prob in zip(boxes, pred_cls_idx, pred_cls_prob):
        x, y, w, h = box
        x0 = x - w / 2
        y0 = y - h / 2
        rect = patches.Rectangle((x0, y0), w, h, linewidth=linewidth, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        idx = idx.item() if hasattr(idx, "item") else ""
        prob = prob.item() if hasattr(prob, "item") else None
        ax.text(
            x0,
            y0,
            f"[{idx}|{prob:.2f}]" if prob is not None else f"[{idx}]",
            color=color,
            fontsize=25,
            ha="left",
            va="top",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
        )


def plot_img_with_boxes(
    image,
    gt_boxes=None,
    gt_class=None,
    pred_boxes=None,
    pred_cls_idx=None,
    pred_cls_prob=None,
    output_path="tmp",
    alpha=1,
    return_figure=False,
    pad=0.02,
):
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    image = utils.to_cpu(image)

    # plot img
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, alpha=alpha)
    ax.axis("off")

    # plot gt_boxes
    if gt_boxes is not None:
        gt_boxes = utils.to_cpu(convert_vertex_to_xyhw(gt_boxes))
        if gt_class is not None:
            gt_class = torch.tensor(utils.to_cpu(gt_class))
        plot_boxes(gt_boxes, ax, pred_cls_idx=gt_class, color="lime")

    # plot preds
    if pred_boxes is not None and len(pred_boxes) > 0:
        pred_boxes = utils.to_cpu(convert_vertex_to_xyhw(pred_boxes))
        if pred_cls_idx is not None:
            pred_cls_idx = torch.tensor(utils.to_cpu(pred_cls_idx))
        if pred_cls_prob is not None:
            pred_cls_prob = torch.tensor(utils.to_cpu(pred_cls_prob))
        plot_boxes(pred_boxes, ax, pred_cls_idx, pred_cls_prob, color="red")

    if return_figure:
        img = get_figure(fig)
        img = img.transpose(2, 0, 1)
        return img
    else:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=pad)
        plt.close(fig)

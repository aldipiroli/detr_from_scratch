import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

import detr.utils.misc as utils
from detr.dataset.voc_dataset import IDX_TO_CLASS

QUERY_COLORS = [
    (1.0, 0.6, 0.1),  # orange
    (1.0, 0.1, 1.0),  # magenta
    (0.2, 0.4, 1.0),  # blue
    (1.0, 0.2, 0.2),  # red
    (1.0, 1.0, 0.2),  # yellow
    (1.0, 0.4, 0.7),  # pink
    (0.0, 1.0, 1.0),  # cyan
]


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


def plot_boxes(boxes, ax, pred_cls_idx=None, pred_cls_prob=None, colors=None, linewidth=5):
    if pred_cls_idx is None:
        pred_cls_idx = [None] * len(boxes)
    if pred_cls_prob is None:
        pred_cls_prob = [None] * len(boxes)

    for i, (box, idx, prob) in enumerate(zip(boxes, pred_cls_idx, pred_cls_prob)):
        x, y, w, h = box
        x0 = x - w / 2
        y0 = y - h / 2
        color = colors[i] if colors is not None else "lime"

        rect = patches.Rectangle((x0, y0), w, h, linewidth=linewidth, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        idx = idx.item() if hasattr(idx, "item") else ""
        prob = prob.item() if hasattr(prob, "item") else None
        ax.text(
            x0,
            y0,
            f"[{IDX_TO_CLASS[idx]}|{prob:.2f}]" if prob is not None else f"[{IDX_TO_CLASS[idx]}]",
            color=color,
            fontsize=25,
            ha="left",
            va="top",
            bbox=dict(facecolor="black", alpha=0.9, edgecolor="none", pad=1),
        )


def plot_att_weights(ax, att_weights, image, query_colors, alpha_scale=1):
    num_queries = att_weights.shape[0]
    H, W = image.shape[0], image.shape[1]
    n_patches = int(att_weights.shape[1] ** 0.5)
    for q in range(num_queries):
        att = att_weights[q].detach().cpu()
        att_map = att.reshape(n_patches, n_patches).numpy()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-6)
        att_up = np.array(Image.fromarray(att_map).resize((W, H), resample=Image.BILINEAR))
        att_alpha = att_up * alpha_scale

        color = np.array(query_colors[q % len(query_colors)])
        color_img = np.ones((H, W, 3)) * color.reshape(1, 1, 3)
        ax.imshow(color_img, alpha=att_alpha)


def plot_img_with_boxes(
    image,
    gt_boxes=None,
    gt_class=None,
    pred_boxes=None,
    pred_cls_idx=None,
    pred_cls_prob=None,
    att_weights=None,
    plot_attention=False,
    fixed_color=False,
    output_path="tmp.png",
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
        plot_boxes(gt_boxes, ax, pred_cls_idx=gt_class)

    # plot pred_boxes
    if pred_boxes is not None and len(pred_boxes) > 0:
        pred_boxes = utils.to_cpu(convert_vertex_to_xyhw(pred_boxes))
        if pred_cls_idx is not None:
            pred_cls_idx = torch.tensor(utils.to_cpu(pred_cls_idx))
        if pred_cls_prob is not None:
            pred_cls_prob = torch.tensor(utils.to_cpu(pred_cls_prob))

        n = len(pred_boxes)
        pred_colors = ["red"] * n if fixed_color else [QUERY_COLORS[i % len(QUERY_COLORS)] for i in range(n)]
        plot_boxes(pred_boxes, ax, pred_cls_idx, pred_cls_prob, colors=pred_colors)

    if att_weights is not None and len(att_weights) > 0 and plot_attention:
        plot_att_weights(ax, att_weights, image, query_colors=pred_colors)

    if return_figure:
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img.transpose(2, 0, 1)
    else:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=pad)
        plt.close(fig)

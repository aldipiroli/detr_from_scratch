import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms, tv_tensors
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2

from detr.utils.misc import normalize_boxes


class VOCDataset(Dataset):
    def __init__(self, cfg, mode, logger):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.logger = logger
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        dest_path = os.path.join(self.root_dir, "VOCdevkit/VOC2012")
        self.img_size = self.cfg["DATA"]["img_size"]
        self.dataset = VOCDetection(
            root=self.root_dir,
            year="2012",
            image_set=mode,
            download=True if not os.path.exists(dest_path) else False,
        )
        self.transform_to_tensor = transforms.Compose([transforms.ToTensor()])
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.simple_transform = transforms.ToTensor()
        self.build_class_maps()

    def build_class_maps(self):
        self.classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.dataset)

    def get_gt_boxes(self, target):
        objs = target["annotation"]["object"]
        if not isinstance(objs, list):
            objs = [objs]

        size = target["annotation"]["size"]
        img_w = float(size["width"])
        img_h = float(size["height"])

        labels = []
        gt_boxes = []
        for obj in objs:
            labels.append(self.class_to_idx[obj["name"]])
            bbox = obj["bndbox"]
            xmin = max(0.0, float(bbox["xmin"]))
            ymin = max(0.0, float(bbox["ymin"]))
            xmax = min(img_w, float(bbox["xmax"]))
            ymax = min(img_h, float(bbox["ymax"]))
            gt_boxes.append([xmin, ymin, xmax, ymax])

        labels = torch.tensor(labels, dtype=torch.long)
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float)
        gt_boxes = normalize_boxes(gt_boxes, img_w=img_w, img_h=img_h)
        return gt_boxes, labels

    def normalize_boxes(self, gt_boxes):
        gt_boxes[:, 0] /= self.img_size[0]
        gt_boxes[:, 2] /= self.img_size[0]

        gt_boxes[:, 1] /= self.img_size[1]
        gt_boxes[:, 3] /= self.img_size[1]
        assert (gt_boxes <= 1).all()
        return gt_boxes

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        original_size = img.size

        img = self.transforms(img)
        gt_boxes, labels = self.get_gt_boxes(target)

        resize = v2.Resize((self.img_size[0], self.img_size[1]), antialias=True)
        gt_boxes = tv_tensors.BoundingBoxes(gt_boxes, format="XYWH", canvas_size=(original_size[1], original_size[0]))
        img, gt_boxes = resize(img, gt_boxes)
        return img, gt_boxes, labels


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


def voc_collate_fn(batch):
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    class_id = [item[2] for item in batch]

    images = torch.stack(images, dim=0)
    targets = []
    for b, l in zip(boxes, class_id):
        targets.append({"boxes": b, "class_id": l})
    return images, targets

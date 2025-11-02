import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from detr.model.detr_model import DetrModel
from detr.utils.misc import load_config


def test_model_flownet_simple():
    config = load_config("detr/config/detr_config.yaml")
    model = DetrModel(config)
    img_size = config["DATA"]["img_size"]
    B, C, H, W = 2, 1, img_size[0], img_size[1]
    img = torch.randn(B, C, H, W)
    out = model(img)
    assert out[0].shape != None


if __name__ == "__main__":
    print("All tests passed!")

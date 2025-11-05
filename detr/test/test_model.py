import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from detr.model.model import DetrModel
from detr.utils.misc import load_config


def test_model():
    config = load_config("detr/config/config.yaml")
    model = DetrModel(config)
    img_size = config["DATA"]["img_size"]
    dec_n_queries = config["MODEL"]["dec_n_queries"]
    B, C, H, W = 2, 3, img_size[0], img_size[1]
    img = torch.randn(B, C, H, W)
    boxes, cls = model(img)
    assert boxes.shape == (B, dec_n_queries, 4)
    assert cls.shape == (B, dec_n_queries, 1)


if __name__ == "__main__":
    print("All tests passed!")

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from detr.model.model import DetrModel, EncoderBlock, Encoder
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

def test_encoder_block():
    embed_dim = 64
    n_heads = 2
    B = 8
    n_patches = 72

    encoder_block = EncoderBlock(embed_dim, n_heads)

    img_feats = torch.randn(B, n_patches, embed_dim)
    pos_encodings = torch.rand(B, n_patches, embed_dim)
    out = encoder_block(img_feats, pos_encodings)
    assert out.shape == (B, n_patches, embed_dim)


def test_encoder_layers():
    embed_dim = 64
    n_heads = 2
    B = 8
    n_patches = 72
    n_layers = 2

    encoder = Encoder(embed_dim, n_heads, n_layers)

    img_feats = torch.randn(B, n_patches, embed_dim)
    pos_encodings = torch.rand(B, n_patches, embed_dim)
    out = encoder(img_feats, pos_encodings)
    assert out.shape == (B, n_patches, embed_dim)


if __name__ == "__main__":
    print("All tests passed!")

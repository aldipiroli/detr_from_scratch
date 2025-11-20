import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from detr.model.model import Decoder, DecoderBlock, DetrModel, DetrModelDYI, Encoder, EncoderBlock
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


def test_model_dyi():
    config = load_config("detr/config/config.yaml")
    model = DetrModelDYI(config)
    img_size = config["DATA"]["img_size"]
    dec_n_queries = config["MODEL"]["dec_n_queries"]
    n_patches = config["MODEL"]["n_patches"]
    n_classes = config["MODEL"]["n_classes"]
    B, C, H, W = 2, 3, img_size[0], img_size[1]
    img = torch.randn(B, C, H, W)
    boxes, cls, attn_weights = model(img)
    assert boxes.shape == (B, dec_n_queries, 4)
    assert cls.shape == (B, dec_n_queries, n_classes + 1)
    assert attn_weights.shape == (B, dec_n_queries, n_patches)


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


def test_decoder_block():
    embed_dim = 64
    n_heads = 2
    B = 8
    n_patches = 72
    n_queries = 10

    decoder_block = DecoderBlock(embed_dim, n_heads)

    queries = torch.randn(B, n_queries, embed_dim)
    queries_pose_embedm = torch.randn(B, n_queries, embed_dim)
    pos_encodings = torch.rand(B, n_patches, embed_dim)
    memory = torch.rand(B, n_patches, embed_dim)
    out, cross_att_weights = decoder_block(queries, queries_pose_embedm, pos_encodings, memory)
    assert out.shape == (B, n_queries, embed_dim)
    assert cross_att_weights.shape == (B, n_queries, n_patches)


def test_decoder_layers():
    embed_dim = 64
    n_heads = 2
    B = 8
    n_patches = 72
    n_layers = 2
    n_queries = 10

    decoder = Decoder(embed_dim, n_heads, n_layers)

    memory = torch.randn(B, n_patches, embed_dim)
    pos_encodings = torch.rand(B, n_patches, embed_dim)
    queries = torch.randn(B, n_queries, embed_dim)
    queries_pose_embedm = torch.randn(B, n_queries, embed_dim)

    out, cross_att_weights = decoder(queries, queries_pose_embedm, memory, pos_encodings)
    assert out.shape == (B, n_queries, embed_dim)
    assert cross_att_weights.shape == (B, n_queries, n_patches)


if __name__ == "__main__":
    print("All tests passed!")

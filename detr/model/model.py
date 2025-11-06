import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=False, normalize_output=True):
        super().__init__()
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        resnet = models.resnet18(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config["DATA"]["img_size"]


class DetrModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.d = config["MODEL"]["d"]
        self.n_patches = config["MODEL"]["n_patches"]
        self.enc_n_heads = config["MODEL"]["enc_n_heads"]
        self.enc_n_layers = config["MODEL"]["enc_n_layers"]
        self.dec_n_heads = config["MODEL"]["dec_n_heads"]
        self.dec_n_layers = config["MODEL"]["dec_n_layers"]
        self.dec_n_queries = config["MODEL"]["dec_n_queries"]

        # Backbone
        self.backbone = ResNet18Backbone()
        self.feat_reduction = nn.Conv2d(512, self.d, 1)

        # Encoder
        self.positional_embeddings = nn.Parameter(torch.rand(1, self.d, self.n_patches))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=self.enc_n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.enc_n_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d, nhead=self.dec_n_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.dec_n_layers)
        self.obj_queries = nn.Parameter(torch.rand(self.dec_n_queries, 1, self.d))

        # FC layer
        self.box_head = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, 4))
        self.cls_head = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, 1))

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.feat_reduction(x)
        x = x.flatten(2, 3)  # B, d, HW
        x = x + self.positional_embeddings.repeat(B, 1, 1)
        x = x.permute(2, 0, 1)  # HW, B, d
        x = self.transformer_encoder(x)

        queries = self.obj_queries.repeat(1, B, 1)
        x = self.transformer_decoder(queries, x)  # n_queries, B, d
        x = x.permute(1, 0, 2)  # B, n_queries, d

        boxes = self.box_head(x)
        cls = self.cls_head(x)
        return boxes, cls


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.K = nn.Linear(embed_dim, embed_dim)
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, pos_encodings):
        x_in = x + pos_encodings
        x_att, _ = self.mhsa(x_in, x_in, x_in)
        x_mid = x_att + x_in
        x_mid = self.layer_norm1(x_mid)
        x_ffn = self.ffn(x_mid)
        x_out = x_ffn + x_mid
        x_out = self.layer_norm2(x_out)
        return x_out


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embed_dim, n_heads) for _ in range(n_layers)])

    def forward(self, x, pos_encodings):
        for block in self.encoder_blocks:
            x = block(x, pos_encodings)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.K = nn.Linear(embed_dim, embed_dim)
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.mhsa1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mhsa2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, queries, pos_encodings, memory):
        x_att, _ = self.mhsa1(queries, queries, queries)
        x_mid = x_att + queries
        x_mid = self.layer_norm1(x_mid)

        k2 = memory + pos_encodings
        v2 = memory
        q2 = queries + x_mid
        x_att2, _ = self.mhsa2(q2, k2, v2)

        x_mid2 = x_att2 + x_mid
        x_mid2 = self.layer_norm2(x_mid2)

        x_ffn = self.ffn(x_mid2)
        x_ffn = x_ffn + x_mid2
        x_ffn = self.layer_norm3(x_ffn)
        return x_ffn


class Decoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim, n_heads) for _ in range(n_layers)])

    def forward(self, queries, memory, pos_encodings):
        for block in self.decoder_blocks:
            queries = block(queries, memory, pos_encodings)
        return queries


class DetrModelDYI(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.d = config["MODEL"]["d"]
        self.n_patches = config["MODEL"]["n_patches"]
        self.enc_n_heads = config["MODEL"]["enc_n_heads"]
        self.enc_n_layers = config["MODEL"]["enc_n_layers"]
        self.dec_n_heads = config["MODEL"]["dec_n_heads"]
        self.dec_n_layers = config["MODEL"]["dec_n_layers"]
        self.dec_n_queries = config["MODEL"]["dec_n_queries"]
        self.n_classes = config["MODEL"]["n_classes"]

        self.backbone = ResNet18Backbone()
        self.feat_reduction = nn.Conv2d(512, self.d, 1)

        self.encoder = Encoder(self.d, self.enc_n_heads, self.enc_n_layers)
        self.decoder = Decoder(self.d, self.dec_n_heads, self.dec_n_layers)
        self.positional_embeddings = nn.Parameter(torch.rand(1, self.n_patches, self.d))
        self.obj_queries = nn.Parameter(torch.rand(1, self.dec_n_queries, self.d))

        self.box_head = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, 4))
        self.cls_head = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, self.n_classes))

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.feat_reduction(x)
        x = x.flatten(2, 3)  # B, d, HW
        x = x.permute(0, 2, 1)
        pose_embed = self.positional_embeddings.repeat(B, 1, 1)
        memory = self.encoder(x, pose_embed)

        queries = self.obj_queries.repeat(B, 1, 1)
        x = self.decoder(queries, memory, pose_embed)

        boxes = self.box_head(x)
        cls = self.cls_head(x)
        return boxes, cls

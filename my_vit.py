import torch
from torch import nn


class PatchEasy(nn.Module):
    def __init__(self):
        super(PatchEasy, self).__init__()
        self.conv1 = nn.Conv1d(12, 96, 8, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2)
        return x


class Vit(nn.Module):
    def __init__(self, num_layers=4, embed_dim=96, n_head=8, num_classes=9):
        super(Vit, self).__init__()
        self.patch_embed = PatchEasy()
        num_patches = 156

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, activation="gelu",
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


if __name__ == "__main__":
    x = torch.randn(32, 12, 1250)
    model = Vit()
    x = model(x)
    print(x.shape)
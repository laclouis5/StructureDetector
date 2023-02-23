import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


class Network(nn.Module):
    def __init__(self, args, pretrained=True):
        super().__init__()

        self.label_count = len(args.labels)  # M
        self.part_count = len(args.parts)  # N
        self.out_channels = self.label_count + self.part_count + 4  # M+N+4
        self.nb_hm = self.label_count + self.part_count  # M+N
        dr = args.down_ratio
        self.out_size = (int(args.width // dr), int(args.height // dr))

        mobilenet = lraspp_mobilenet_v3_large(
            num_classes=self.out_channels,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,
        )

        self.net = torch.nn.Sequential(mobilenet.backbone, mobilenet.classifier)

    def forward(self, x):  # (B, 3, H, W)
        out = self.net(x)  # (B, M+N+4, H/8, W/8)
        out = F.interpolate(  # (B, M+N+4, H/4, W/4)
            out, size=self.out_size, align_corners=False, mode="bilinear"
        )

        return {  # R = 4
            "anchor_hm": out[:, : self.label_count],  # (B, M, H/R, W/R)
            "part_hm": out[:, self.label_count : self.nb_hm],  # (B, N, H/R, W/R)
            "offsets": out[:, self.nb_hm : (self.nb_hm + 2)],  # (B, 2, H/R, W/R)
            "embeddings": out[:, (self.nb_hm + 2) : (self.nb_hm + 4)],
        }  # (B, 2, H/R, W/R)

    def save(self, path="last_model.pth"):
        torch.save(self.state_dict(), path)

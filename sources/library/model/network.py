import torch
import torch.nn as nn
import torchvision
from mmdet.models import Darknet


class Fpn(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, input, shortcut):
        return self.conv(self.up(input) + self.lateral(shortcut))


class Head(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        return self.conv(input)


class Network(nn.Module):
    
    def __init__(self, args, pretrained=True):
        super().__init__()

        self.label_count = len(args.labels)  # M
        self.part_count = len(args.parts)  # N
        self.out_channels = self.label_count + self.part_count + 4  # M+N+4
        self.fpn_depth = args.fpn_depth

        self.darknet = Darknet(out_indices=(2, 3, 4, 5))

        self.up1 = nn.Conv2d(1024, self.fpn_depth, kernel_size=1)  # x1 -> /32
        self.up2 = Fpn(512, self.fpn_depth)  # x2 -> /16
        self.up3 = Fpn(256, self.fpn_depth)  # x2 -> /8
        self.up4 = Fpn(128, self.fpn_depth)  # x2 -> /4

        self.head = Head(self.fpn_depth, self.out_channels)

    def forward(self, x):  # (B, 3, H, W)
        p2, p3, p4, p5 = self.darknet(x)

        f4 = self.up1(p5)  # (B, 128, H/32, W/32)
        f3 = self.up2(f4, p4)  # (B, 128, H/16, W/16)
        f2 = self.up3(f3, p3)  # (B, 128, H/8, W/8)
        f1 = self.up4(f2, p2)  # (B, 128, H/4, W/4)

        out = self.head(f1)  # (B, M+N+4, H/4, W/4)

        nb_hm = self.label_count + self.part_count  # M+N

        return {  # R = 4
            "anchor_hm": out[:, :self.label_count],  # (B, M, H/R, W/R)
            "part_hm": out[:, self.label_count:nb_hm],  # (B, N, H/R, W/R)
            "offsets": out[:, nb_hm:(nb_hm + 2)],  # (B, 2, H/R, W/R)
            "embeddings": out[:, (nb_hm + 2):(nb_hm + 4)]}  # (B, 2, H/R, W/R)

    def save(self, path="last_model.pth"):
        torch.save(self.state_dict(), path)
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


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


class ESCPN(nn.Module):

    def __init__(self, in_channels, out_channels: int, scale: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * scale**2, kernel_size=1)
        self.up = nn.PixelShuffle(scale)

    def forward(self, input):
        return self.up(self.conv(input))


class Network(nn.Module):
    
    def __init__(self, args, pretrained=True):
        super().__init__()

        self.label_count = len(args.labels)  # M
        self.part_count = len(args.parts)  # N
        self.out_channels = self.label_count + self.part_count + 4  # M+N+4
        self.fpn_depth = args.fpn_depth

        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        self.adpater = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # /4 -> /4

        self.down1 = resnet.layer1  # /1 -> /4
        self.down2 = resnet.layer2  # /2 -> /8
        self.down3 = resnet.layer3  # /2 -> /16
        self.down4 = resnet.layer4  # /2 -> /32

        self.head = ESCPN(512, self.out_channels, scale=8)  # x8 -> /4

    def forward(self, x):  # (B, 3, H, W)
        p1 = self.adpater(x)  # (B, 64, H/4, W/4)

        p2 = self.down1(p1)  # (B, 64, H/4, W/4)
        p3 = self.down2(p2)  # (B, 128, H/8, W/8)
        p4 = self.down3(p3)  # (B, 256, H/16, W/16)
        p5 = self.down4(p4)  # (B, 512, H/32, W/32)

        out = self.head(p5)  # (B, N+M+4, H/4, W/4)

        nb_hm = self.label_count + self.part_count  # M+N

        return {  # R = 4
            "anchor_hm": out[:, :self.label_count],  # (B, M, H/R, W/R)
            "part_hm": out[:, self.label_count:nb_hm],  # (B, N, H/R, W/R)
            "offsets": out[:, nb_hm:(nb_hm + 2)],  # (B, 2, H/R, W/R)
            "embeddings": out[:, (nb_hm + 2):(nb_hm + 4)]}  # (B, 2, H/R, W/R)

    def save(self, path="last_model.pth"):
        torch.save(self.state_dict(), path)
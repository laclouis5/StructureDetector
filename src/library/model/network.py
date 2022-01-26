from numpy import short
import torch
import torch.nn as nn
import torchvision

from library.utils.utils import clamped_sigmoid


class Fpn(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, input: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(input) + self.lateral(shortcut))


class FPNTop(Fpn):

    def forward(self, input: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        input = self.up(input)
        shortcut = self.lateral(self.up(shortcut))
        return self.conv(input + shortcut)


class Head(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class Network(nn.Module):
    
    def __init__(self, args, pretrained=True):
        super().__init__()

        self.label_count = len(args.labels)  # M
        self.out_channels = self.label_count + 4  # M + 4
        self.fpn_depth = args.fpn_depth

        backbone = torchvision.models.resnet34(pretrained=pretrained)

        self.adapter = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)  # /4 -> /4

        self.down1 = backbone.layer1  # /1 -> /4
        self.down2 = backbone.layer2  # /2 -> /8
        self.down3 = backbone.layer3  # /2 -> /16
        self.down4 = backbone.layer4  # /2 -> /32

        self.up1 = nn.Conv2d(512, self.fpn_depth, kernel_size=1)  # x1 -> /32
        self.up2 = Fpn(256, self.fpn_depth)  # x2 -> /16
        self.up3 = Fpn(128, self.fpn_depth)  # x2 -> /8
        self.up4 = Fpn(64, self.fpn_depth)  # x2 -> /4

        # self.up5 = FPNTop(64, self.fpn_depth)  # x2 -> /2

        self.head = Head(self.fpn_depth, self.out_channels)

    def forward(self, x: torch.Tensor) -> dict[torch.Tensor]:  # (B, 3, H, W)
        p1 = self.adapter(x)  # (B, 64, H/4, W/4)

        p2 = self.down1(p1)  # (B, 64, H/4, W/4)
        p3 = self.down2(p2)  # (B, 128, H/8, W/8)
        p4 = self.down3(p3)  # (B, 256, H/16, W/16)
        p5 = self.down4(p4)  # (B, 512, H/32, W/32)

        f4 = self.up1(p5)  # (B, 128, H/32, W/32)
        f3 = self.up2(f4, p4)  # (B, 128, H/16, W/16)
        f2 = self.up3(f3, p3)  # (B, 128, H/8, W/8)
        f1 = self.up4(f2, p2)  # (B, 128, H/4, W/4)

        # f0 = self.up5(f1, p1)  # (B, 64, H/2, W/2)
        # out = self.head(f0)  # (B, M+4, H/2, W/2)

        out = self.head(f1)  # (B, M+4, H/4, W/4)

        nb_hm = self.label_count  # M

        return {  # R = 4
            "heatmaps": clamped_sigmoid(out[:, :self.label_count]),  # (B, M, H/R, W/R)
            "offsets": out[:, nb_hm:(nb_hm + 2)],  # (B, 2, H/R, W/R)
            "embeddings": out[:, (nb_hm + 2):(nb_hm + 4)]}  # (B, 2, H/R, W/R)

    def save(self, path="last_model.pth"):
        torch.save(self.state_dict(), path)


class NetworkRegNet(nn.Module):
    
    def __init__(self, args, pretrained=True):
        super().__init__()

        self.label_count = len(args.labels)  # M
        self.out_channels = self.label_count + 4  # M + 4
        self.fpn_depth = args.fpn_depth

        backbone = torchvision.models.regnet_x_1_6gf(pretrained=pretrained)
        stages = backbone.trunk_output

        self.adapter = backbone.stem # /4 -> /4

        self.down1 = stages.block1  # /1 -> /4
        self.down2 = stages.block2  # /2 -> /8
        self.down3 = stages.block3  # /2 -> /16
        self.down4 = stages.block4  # /2 -> /32

        self.up1 = nn.Conv2d(912, self.fpn_depth, kernel_size=1)  # x1 -> /32
        self.up2 = Fpn(408, self.fpn_depth)  # x2 -> /16
        self.up3 = Fpn(168, self.fpn_depth)  # x2 -> /8
        self.up4 = Fpn(72, self.fpn_depth)  # x2 -> /4

        self.head = Head(self.fpn_depth, self.out_channels)

    def forward(self, x: torch.Tensor) -> dict[torch.Tensor]:  # (B, 3, H, W)
        p1 = self.adapter(x)  # (B, 64, H/4, W/4)

        p2 = self.down1(p1)  # (B, 64, H/4, W/4)
        p3 = self.down2(p2)  # (B, 128, H/8, W/8)
        p4 = self.down3(p3)  # (B, 256, H/16, W/16)
        p5 = self.down4(p4)  # (B, 512, H/32, W/32)

        f4 = self.up1(p5)  # (B, 128, H/32, W/32)
        f3 = self.up2(f4, p4)  # (B, 128, H/16, W/16)
        f2 = self.up3(f3, p3)  # (B, 128, H/8, W/8)
        f1 = self.up4(f2, p2)  # (B, 128, H/4, W/4)

        out = self.head(f1)  # (B, M+4, H/4, W/4)

        nb_hm = self.label_count  # M

        return {  # R = 4
            "heatmaps": clamped_sigmoid(out[:, :self.label_count]),  # (B, M, H/R, W/R)
            "offsets": out[:, nb_hm:(nb_hm + 2)],  # (B, 2, H/R, W/R)
            "embeddings": out[:, (nb_hm + 2):(nb_hm + 4)]}  # (B, 2, H/R, W/R)

    def save(self, path="last_model.pth"):
        torch.save(self.state_dict(), path)
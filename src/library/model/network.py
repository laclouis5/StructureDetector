import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet34, ResNet34_Weights


class Fpn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input, shortcut):
        sum_ = self.up(input) + self.lateral(shortcut)
        return self.conv(sum_) + sum_


class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, input):
        return self.conv(input)


class ASPP(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilations: list[int] = [2, 4, 8]
    ):
        super().__init__()

        self.aspp_blocks = torch.nn.ModuleList(
            self.make_aspp_block(in_channels, out_channels, dilation)
            for dilation in dilations
        )

        self.conv = nn.Conv2d(
            len(dilations) * out_channels, out_channels, kernel_size=1
        )

        self._init_weights()

    @staticmethod
    def make_aspp_block(in_channels: int, out_channels: int, dilation: int):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # (B, C, H, W)
        outputs = [block(x) for block in self.aspp_blocks]  # n * (B, O, H, W)
        out = torch.cat(outputs, dim=1)  # (B, n*O, H, W)
        return self.conv(out)  # (B, O, H, W)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:  # (B, C, H, W)
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)  # (B, C)
        y = self.fc(y).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * y  # (B, C, H, W)


class Network(nn.Module):
    def __init__(self, args, pretrained=True, raw_output: bool = False):
        super().__init__()
        self.raw_output = raw_output
        self.label_count = len(args.labels)  # M
        self.part_count = len(args.parts)  # N
        self.out_channels = self.label_count + self.part_count + 4  # M+N+4
        self.fpn_depth = args.fpn_depth

        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        self.adpater = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # /4 -> /4

        self.down1 = resnet.layer1  # /1 -> /4
        self.down2 = resnet.layer2  # /2 -> /8
        self.down3 = resnet.layer3  # /2 -> /16
        self.down4 = resnet.layer4  # /2 -> /32

        self.sqex1 = SqueezeExciteBlock(64)
        self.sqex2 = SqueezeExciteBlock(64)
        self.sqex3 = SqueezeExciteBlock(128)
        self.sqex4 = SqueezeExciteBlock(256)

        self.bridge = ASPP(512, self.fpn_depth)  # x1 -> /32

        self.up2 = Fpn(256, self.fpn_depth)  # x2 -> /16
        self.up3 = Fpn(128, self.fpn_depth)  # x2 -> /8
        self.up4 = Fpn(64, self.fpn_depth)  # x2 -> /4

        self.head = Head(self.fpn_depth, self.out_channels)

    def forward(self, x):  # (B, 3, H, W)
        p1 = self.adpater(x)  # (B, 64, H/4, W/4)

        p2 = self.sqex1(p1)
        p2 = self.down1(p2)  # (B, 64, H/4, W/4)
        
        p3 = self.sqex2(p2)
        p3 = self.down2(p3)  # (B, 128, H/8, W/8)
        
        p4 = self.sqex3(p3)
        p4 = self.down3(p4)  # (B, 256, H/16, W/16)
        
        p5 = self.sqex4(p4)
        p5 = self.down4(p5)  # (B, 512, H/32, W/32)

        f4 = self.bridge(p5)  # (B, 128, H/32, W/32)

        f3 = self.up2(f4, p4)  # (B, 128, H/16, W/16)
        f2 = self.up3(f3, p3)  # (B, 128, H/8, W/8)
        f1 = self.up4(f2, p2)  # (B, 128, H/4, W/4)

        out = self.head(f1)  # (B, M+N+4, H/4, W/4)

        if self.raw_output:
            return out

        nb_hm = self.label_count + self.part_count  # M+N

        return {  # R = 4
            "anchor_hm": out[:, : self.label_count],  # (B, M, H/R, W/R)
            "part_hm": out[:, self.label_count : nb_hm],  # (B, N, H/R, W/R)
            "offsets": out[:, nb_hm : (nb_hm + 2)],  # (B, 2, H/R, W/R)
            "embeddings": out[:, (nb_hm + 2) : (nb_hm + 4)],
        }  # (B, 2, H/R, W/R)

    def save(self, path="last_model.pth"):
        torch.save(self.state_dict(), path)

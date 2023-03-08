import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.upscale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x2 = self.upscale(x2)

        diff_y = x1.size()[2] - x2.size()[2]
        diff_x = x1.size()[3] - x2.size()[3]

        x2 = F.pad(
            x2,
            [
                diff_x // 2,
                diff_x - diff_y // 2,
                diff_y // 2,
                diff_y - diff_x // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


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

        self.conv1 = DoubleConv(3, 64)

        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024)

        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)

        self.head = Head(64, self.out_channels)

    def forward(self, x):  # (B, 3, H, W)
        p1 = self.conv1(x)  # (B, 64, H/4, W/4)

        p2 = self.down1(p1)  # (B, 64, H/4, W/4)
        p3 = self.down2(p2)  # (B, 128, H/8, W/8)
        p4 = self.down3(p3)  # (B, 256, H/16, W/16)
        p5 = self.down4(p4)  # (B, 512, H/32, W/32)

        f4 = self.up1(p4, p5)  # (B, 128, H/32, W/32)
        f3 = self.up2(p3, f4)  # (B, 128, H/16, W/16)
        f2 = self.up3(p2, f3)  # (B, 128, H/8, W/8)
        f1 = self.up4(p1, f2)  # (B, 128, H/4, W/4)

        out = self.head(f1)  # (B, M+N+4, H/4, W/4)

        nb_hm = self.label_count + self.part_count  # M+N

        return {  # R = 4
            "anchor_hm": out[:, : self.label_count],  # (B, M, H/R, W/R)
            "part_hm": out[:, self.label_count : nb_hm],  # (B, N, H/R, W/R)
            "offsets": out[:, nb_hm : (nb_hm + 2)],  # (B, 2, H/R, W/R)
            "embeddings": out[:, (nb_hm + 2) : (nb_hm + 4)],
        }  # (B, 2, H/R, W/R)

    def save(self, path="last_model.pth"):
        torch.save(self.state_dict(), path)

import torch
from torch import nn


def conv1x1(in_channels: int, out_channels: int, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
    )


def conv3x3(
    in_channels: int, out_channels: int, stride: int = 1, bias: bool = False
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
    )


class Conv1x1Block(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            conv1x1(in_channels=in_channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )


class Conv3x3Block(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )


class ConvRes(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            Conv3x3Block(in_channels=channels, out_channels=channels),
            conv3x3(in_channels=channels, out_channels=channels),
            nn.BatchNorm2d(num_features=channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer(input)
        return self.relu(x + input)


class Bottleneck(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        inter_channels = channels // expansion

        self.layer = nn.Sequential(
            Conv1x1Block(in_channels=channels, out_channels=inter_channels),
            Conv3x3Block(in_channels=inter_channels, out_channels=inter_channels),
            conv1x1(in_channels=inter_channels, out_channels=channels),
            nn.BatchNorm2d(num_features=channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer(input)
        return self.relu(x + input)


class ExpandedBottleneck(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        out_channels = channels * expansion

        self.layer = nn.Sequential(
            Conv1x1Block(in_channels=channels, out_channels=channels),
            Conv3x3Block(in_channels=channels, out_channels=channels),
            conv1x1(in_channels=channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.expand = nn.Sequential(
            conv1x1(in_channels=channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.layer(input)
        residual = self.expand(input)
        return self.relu(x + residual)


class Up(nn.Sequential):
    def __init__(self, d: int, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2**d, mode="bilinear", align_corners=True),
            conv1x1(in_channels=in_channels, out_channels=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
        )


class Down(nn.Sequential):
    def __init__(self, d: int, in_channels: int, out_channels: int) -> None:
        layers = [
            Conv3x3Block(in_channels=in_channels, out_channels=in_channels, stride=2)
            for _ in range(d - 1)
        ]

        last_layer = nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=out_channels, stride=2),
            nn.BatchNorm2d(num_features=out_channels),
        )

        super().__init__(*layers, last_layer)


class FusionBranch(nn.Module):
    def __init__(self, r: int, channels: list[int]) -> None:
        super().__init__()

        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList(self.fsr(s=s, r=r) for s in range(len(channels)))

    def fsr(self, s: int, r: int) -> nn.Module:
        c1, c2 = self.channels[s], self.channels[r]

        if s == r:  # Id
            return nn.Identity()
        elif s < r:  # Down
            return Down(d=r - s, in_channels=c1, out_channels=c2)
        else:  # Up
            return Up(d=s - r, in_channels=c1, out_channels=c2)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        outputs = (layer(input) for input, layer in zip(inputs, self.layers))
        return self.relu(sum(outputs))


class Fusion(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            FusionBranch(r=r, channels=channels) for r in range(len(channels))
        )

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(inputs) for layer in self.layers]


class HRBranch(nn.Sequential):
    def __init__(self, channels: int, nb_convs: int):
        super().__init__(*(ConvRes(channels=channels) for _ in range(nb_convs)))


class HRBlock(nn.Module):
    def __init__(self, channels: list[int], nb_convs: list[int]):
        super().__init__()

        self.branches = nn.ModuleList(
            HRBranch(channels=c, nb_convs=n) for c, n in zip(channels, nb_convs)
        )
        self.fusion = Fusion(channels=channels)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = [branch(input) for input, branch in zip(inputs, self.branches)]
        return self.fusion(outputs)


class Transition(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: list[int]):
        super().__init__()

        self.s_in, s_out = len(in_channels), len(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList(self.branch(i) for i in range(s_out))

    def branch(self, i: int) -> nn.Module:
        if i == self.s_in:  # New branch
            return Conv3x3Block(
                in_channels=self.in_channels[-1],
                out_channels=self.out_channels[i],
                stride=2,
            )
        elif self.in_channels[i] != self.out_channels[i]:  # Adapt #channels
            return Conv3x3Block(
                in_channels=self.in_channels[i], out_channels=self.out_channels[i]
            )
        else:  # No adaptation
            return nn.Identity()

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = [layer(input) for input, layer in zip(inputs, self.layers)]
        outputs.append(self.layers[-1](inputs[-1]))
        return outputs


class Stem(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            Conv3x3Block(in_channels=in_channels, out_channels=out_channels, stride=2),
            Conv3x3Block(in_channels=out_channels, out_channels=out_channels, stride=2),
        )


class FirstStage(nn.Sequential):
    def __init__(self, in_channels: int, expansion: int = 4):
        out_channels = expansion * in_channels
        super().__init__(
            ExpandedBottleneck(channels=in_channels, expansion=expansion),
            *(Bottleneck(channels=out_channels, expansion=expansion) for _ in range(3)),
        )

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        return [super().forward(input)]


class Stage(nn.Sequential):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        nb_convs: list[int],
        nb_blocks: int,
    ):
        super().__init__(
            Transition(in_channels=in_channels, out_channels=out_channels),
            *(
                HRBlock(channels=out_channels, nb_convs=nb_convs)
                for _ in range(nb_blocks)
            ),
        )


class HRNet18(nn.Sequential):
    def __init__(self, in_channels: int = 3):
        super().__init__(
            Stem(in_channels=in_channels, out_channels=64),
            FirstStage(in_channels=64),
            Stage(  # 2
                in_channels=[256],
                out_channels=[18, 36],
                nb_convs=[4, 4],
                nb_blocks=1,
            ),
            Stage(  # 3
                in_channels=[18, 36],
                out_channels=[18, 36, 72],
                nb_convs=[4, 4, 4],
                nb_blocks=4,
            ),
            Stage(  # 4
                in_channels=[18, 36, 72],
                out_channels=[18, 36, 72, 144],
                nb_convs=[4, 4, 4, 4],
                nb_blocks=3,
            ),
        )


class HRNet32(nn.Sequential):
    def __init__(self, in_channels: int = 3):
        super().__init__(
            Stem(in_channels=in_channels, out_channels=64),
            FirstStage(in_channels=64),
            Stage(  # 2
                in_channels=[256],
                out_channels=[32, 64],
                nb_convs=[4, 4],
                nb_blocks=1,
            ),
            Stage(  # 3
                in_channels=[32, 64],
                out_channels=[32, 64, 128],
                nb_convs=[4, 4, 4],
                nb_blocks=4,
            ),
            Stage(  # 4
                in_channels=[32, 64, 128],
                out_channels=[32, 64, 128, 256],
                nb_convs=[4, 4, 4, 4],
                nb_blocks=3,
            ),
        )


class HRNet48(nn.Sequential):
    def __init__(self, in_channels: int = 3):
        super().__init__(
            Stem(in_channels=in_channels, out_channels=64),
            FirstStage(in_channels=64),
            Stage(  # 2
                in_channels=[256],
                out_channels=[48, 96],
                nb_convs=[4, 4],
                nb_blocks=1,
            ),
            Stage(  # 3
                in_channels=[48, 96],
                out_channels=[48, 96, 192],
                nb_convs=[4, 4, 4],
                nb_blocks=4,
            ),
            Stage(  # 4
                in_channels=[48, 96, 192],
                out_channels=[48, 96, 192, 384],
                nb_convs=[4, 4, 4, 4],
                nb_blocks=3,
            ),
        )


class HeadV1(nn.Module):
    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        return inputs[0]


class HeadV2(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int) -> None:
        super().__init__()
        channels = sum(in_channels)

        self.layers = nn.ModuleList(
            nn.Upsample(scale_factor=2**d, mode="bilinear", align_corners=True)
            for d in range(1, len(in_channels))
        )
        self.last_layer = nn.Sequential(
            Conv1x1Block(in_channels=channels, out_channels=channels),
            conv1x1(in_channels=channels, out_channels=out_channels, bias=False),
        )

    def forward(self, inputs: list[torch.Tensor]):
        outputs = [inputs[0]] + [
            layer(input) for input, layer in zip(inputs[1:], self.layers)
        ]
        output = torch.cat(outputs, dim=1)
        return self.last_layer(output)


class HRNet18V2(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            HRNet18(in_channels=in_channels),
            HeadV2(in_channels=[18, 36, 72, 144], out_channels=out_channels),
        )


class HRNet18V2Pose(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.label_count = len(args.labels)
        self.part_count = len(args.parts)
        self.out_channels = self.label_count + self.part_count + 4

        self.net = HRNet18V2(in_channels=3, out_channels=self.out_channels)

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        nb_hm = self.out_channels - 4
        out = self.net(input)
        return {  # R = 4
            "anchor_hm": out[:, : self.label_count],  # (B, M, H/R, W/R)
            "part_hm": out[:, self.label_count : nb_hm],  # (B, N, H/R, W/R)
            "offsets": out[:, nb_hm : (nb_hm + 2)],  # (B, 2, H/R, W/R)
            "embeddings": out[:, (nb_hm + 2) : (nb_hm + 4)],
        }  # (B, 2, H/R, W/R)

    def save(self, _):
        ...

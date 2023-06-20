import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from collections import OrderedDict


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()

        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x

        return out


class LRASPPHead(nn.Module):
    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int = 128,
    ) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s

        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


class LRASPPMobileNetV3Large(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        net = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,
            dilated=True,
        )

        self.backbone = IntermediateLayerGetter(
            net.features,
            return_layers={"4": "low", "16": "high"},
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lraspp = LRASPPHead(40, 960, num_classes=num_classes)

    def forward(self, input: Tensor) -> Tensor:
        features = self.backbone(input)
        mask = self.lraspp(features)
        return F.interpolate(
            mask, size=input.shape[-2:], mode="bilinear", align_corners=False
        )


class Network(nn.Module):
    def __init__(self, args, pretrained=True, raw_output: bool = False):
        super().__init__()
        self.raw_output = raw_output
        self.label_count = len(args.labels)  # M
        self.part_count = len(args.parts)  # N
        self.out_channels = self.label_count + self.part_count + 4  # M+N+4

        self.net = LRASPPMobileNetV3Large(
            num_classes=self.out_channels, pretrained=pretrained
        )

    def forward(self, x):  # (B, 3, H, W)
        out = self.net(x)  # (B, M+N+4, H/4, W/4)

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

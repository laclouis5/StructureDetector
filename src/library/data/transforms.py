from ..utils import *
import torchvision.transforms as torchtf
import torchvision.transforms.functional as F
import torch
from PIL.Image import Image as PILImage
from itertools import islice


class RandomHorizontalFlip:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, input: PILImage, target: GraphAnnotation) -> tuple[PILImage, GraphAnnotation]:
        if torch.randn(1).item() < self.prob:
            return F.hflip(input), target.fliped_lr(input.size)
        else:
            return input, target

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(prob: {self.prob})"


class RandomColorJitter:

    def __init__(self, brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05):
        self.transform = torchtf.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

    def __call__(self, input: PILImage, target: GraphAnnotation) -> tuple[PILImage, GraphAnnotation]:
        return self.transform(input), target

    def __repr__(self) -> str:
        return f"RandomColorJitter(brightness: {self.transform.brightness}, contrast: {self.transform.contrast}, saturation: {self.transform.saturation}, hue: {self.transform.hue})"


class Resize:

    """Size is a (width, height) tuple"""

    def __init__(self, size):
        if isinstance(size, int):
            self.width, self.height = (size, size)
        elif isinstance(size, tuple):
            self.width, self.height = size
        else:
            raise IOError("Input 'size' must be an int or a tuple<int>.")

    def __call__(self, input: PILImage, target: GraphAnnotation) -> tuple[PILImage, GraphAnnotation]:
        image = F.resize(input, (self.height, self.width))
        annotation = target.resized(input.size, (self.width, self.height))
        return image, annotation

    def __repr__(self) -> str:
        return f"Resize(width: {self.width}, height: {self.height})"


class RandomResize:

    def __init__(self, args, ratios=None):
        if ratios is None:
            ratios = [1 + 1/16 * ratio for ratio in range(-4, 5)]

        for ratio in ratios:
            assert (ratio * 32) % 32 == 0, "Ratios should resolve to multiple of 32"

        self.ratios = ratios
        self.width = args.width
        self.height = args.height

    def __call__(self, input: PILImage, target: GraphAnnotation) -> tuple[PILImage, GraphAnnotation]:
        ratio = self.ratios[torch.randint(len(self.ratios), (1,)).item()]
        width, height = int(ratio * self.width), int(ratio * self.height)
        image = F.resize(input, (height, width))
        annotation = target.resize(input.size, (width, height))

        return image, annotation

    def __repr__(self) -> str:
        return f"RandomResize(ratios: {self.ratios}, img_width: {self.width}, img_height: {self.height})"


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs) -> tuple[PILImage, GraphAnnotation]:
        for transform in self.transforms:
            inputs = transform(*inputs)

        return inputs

    def __repr__(self) -> str:
        return f"Compose(transforms: {self.transforms})"


class Normalize:

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = torchtf.Normalize(mean=mean, std=std)

    def __call__(self, input: PILImage, target: GraphAnnotation) -> tuple[torch.Tensor, GraphAnnotation]:
        output = F.to_tensor(input)  # input is normalized in [0, 1]
        return self.transform(output), target

    def __repr__(self) -> str:
        return f"Normalize(mean: {self.transform.mean}, std: {self.transform.std})"


class Encode:

    def __init__(self, args):
        self.down_ratio = args.down_ratio
        self.labels = args.labels
        self.parts = args.parts
        self.max_objects = args.max_objects
        self.max_parts = args.max_parts
        self.sigma_gauss = args.sigma_gauss

    def __call__(self, input: PILImage, target: GraphAnnotation) -> dict:
        img_h, img_w = input.shape[-2:]
        out_w, out_h = int(img_w / self.down_ratio), int(img_h / self.down_ratio)
        kp_idx = 0

        sigma = self.sigma_gauss * min(out_w, out_h) / 3
        Y, X = torch.meshgrid(torch.arange(out_h), torch.arange(out_w))

        heatmaps = torch.zeros(len(self.labels) + len(self.parts), out_h, out_w)
        anchor_inds = torch.zeros(self.max_objects, dtype=torch.long)
        anchor_offs = torch.zeros(self.max_objects, 2)
        embeddings = torch.zeros(self.max_parts, 2)
        anchor_mask = torch.zeros(self.max_objects, dtype=torch.bool)

        resized_target = target.resized((img_w, img_h), (out_w, out_h))
        resized_target.clip((out_w, out_h))

        keypoints = islice(resized_target.graph.keypoints, self.max_objects)

        for kp_index, keypoint in enumerate(keypoints):
            label_index = self.labels[keypoint.kind]
            x, y = keypoint.x, keypoint.y
            x_r, y_r = int(x), int(y)
            
            heatmap_kp = gaussian_2d(X, Y, x_r, y_r, sigma)
            heatmaps[label_index] = torch.max(heatmaps[label_index], heatmap_kp)
            anchor_inds[kp_index] = y_r * out_w + x_r

            anchor_offset = torch.tensor((x - x_r, y - y_r))
            anchor_offs[kp_index] = anchor_offset
            anchor_mask[kp_index] = True

        # TODO

        # for obj_idx, obj in enumerate(resized_target.objects[:self.max_objects]):
        #     for kp in obj.parts:
        #         kind_index = self.parts[kp.kind] + len(self.labels)  # index in whole HM

        #         part_hm = gaussian_2d(X, Y, int(kp.x), int(kp.y), sigma)
        #         heatmaps[kind_index] = torch.max(heatmaps[kind_index], part_hm)

        #         parts_inds[kp_idx] = int(kp.y) * out_w + int(kp.x)

        #         part_offset = torch.tensor((kp.x - int(kp.x), kp.y - int(kp.y)))
        #         part_offs[kp_idx] = part_offset

        #         embedding = torch.tensor((obj.x - kp.x, obj.y - kp.y))
        #         embeddings[kp_idx] = embedding

        #         part_mask[kp_idx] = True

        #         kp_idx += 1
        #         if kp_idx == self.max_parts: break

        #     if kp_idx == self.max_parts: break

        return {
            "image": input,
            "anchor_hm": heatmaps[:len(self.labels)],
            "part_hm": heatmaps[len(self.labels):],
            "anchor_inds": anchor_inds, "part_inds": parts_inds,
            "anchor_offsets": anchor_offs, "part_offsets": part_offs,
            "embeddings": embeddings,
            "anchor_mask": anchor_mask, "part_mask": part_mask,
            "annotation": target}

    def __repr__(self) -> str:
        return f"Encode(max_objects: {self.max_objects}, max_parts: {self.max_parts}, down_ratio: {self.down_ratio}, nb_labels: {len(self.labels)}, nb_parts: {len(self.parts)})"


class TrainAugmentation:

    ratios = (0.75, 0.8125, 0.875, 0.9375, 1, 1.0625, 1.125, 1.1875, 1.25)

    def __init__(self, args):
        self.args = args
        self.transform = Compose([
            Resize((args.width, args.height)),
            RandomColorJitter(),
            RandomHorizontalFlip(),
            Normalize(),
            Encode(args),
        ]) if not args.no_augmentation else Compose([
            Resize((args.width, args.height)),
            Normalize(),
            Encode(args),
        ])

    def trigger_random_resize(self):
        if self.args.no_augmentation:
            return

        resize_ratio = self.ratios[torch.randint(len(self.ratios), (1,)).item()]
        width = int(resize_ratio * self.args.width / 32) * 32
        height = int(resize_ratio * self.args.height / 32) * 32
        self.transform.transforms[0] = Resize((width, height))

    def __call__(self, input: PILImage, target: GraphAnnotation) -> dict:
        return self.transform(input, target)

    def __repr__(self) -> str:
        return f"TrainAugmentation(transforms: {self.transform})"


class ValidationAugmentation:

    def __init__(self, args):
        self.transform = Compose([
            Resize((args.width, args.height)),
            Normalize(),
            Encode(args),
        ])

    def __call__(self, input: PILImage, target: GraphAnnotation) -> dict:
        return self.transform(input, target)

    def __repr__(self) -> str:
        return f"ValidationAugmentation(transforms: {self.transform})"


class PredictionTransformation:

    def __init__(self, args):
        self.tf = torchtf.Compose([
            torchtf.Resize((args.height, args.width)),
            torchtf.ToTensor(),
            torchtf.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, input):
        return self.tf(input)

    def __repr__(self):
        return f"PredictionTranformation(tranforms: {self.tf})"
from collections import defaultdict
from pathlib import Path

import torch
import torch.utils.data as data
from PIL import Image

from ..data import ImageAnnotation
from ..utils import files_with_extension
from .transforms import Compose


class CropDataset(data.Dataset):
    def __init__(self, args, directories, transforms=None):
        super().__init__()
        self.args = args

        if isinstance(transforms, list):
            self.transform = Compose(transforms)
        else:
            self.transform = transforms

        if isinstance(directories, (str, Path)):
            self.files = files_with_extension(directories, ".json")
        elif isinstance(directories, list):
            self.files = [
                file
                for directory in directories
                for file in files_with_extension(directory, ".json")
            ]
        else:
            raise ValueError(
                "'directories' should be either a unique directory or a list of directories."
            )

        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        annotation = ImageAnnotation.from_json(self.files[index], self.args.anchor_name)
        image = Image.open(annotation.image_path)
        annotation.img_size = image.size

        if self.transform is not None:
            return self.transform(image, annotation)

        return (image, annotation)

    def localize_image_names(self):
        for file in self.files:
            annotation = ImageAnnotation.from_json(file, self.args.anchor_name)
            annotation.image_path = file.parent / annotation.image_name
            annotation.save_json(file.parent)

    @staticmethod
    def collate_fn(elements):
        return {
            "image": torch.stack([element["image"] for element in elements], dim=0),
            "anchor_hm": torch.stack(
                [element["anchor_hm"] for element in elements], dim=0
            ),
            "part_hm": torch.stack([element["part_hm"] for element in elements], dim=0),
            "anchor_offsets": torch.stack(
                [element["anchor_offsets"] for element in elements], dim=0
            ),
            "part_offsets": torch.stack(
                [element["part_offsets"] for element in elements], dim=0
            ),
            "embeddings": torch.stack(
                [element["embeddings"] for element in elements], dim=0
            ),
            "anchor_inds": torch.stack(
                [element["anchor_inds"] for element in elements], dim=0
            ),
            "part_inds": torch.stack(
                [element["part_inds"] for element in elements], dim=0
            ),
            "anchor_mask": torch.stack(
                [element["anchor_mask"] for element in elements], dim=0
            ),
            "part_mask": torch.stack(
                [element["part_mask"] for element in elements], dim=0
            ),
            "annotation": [element["annotation"] for element in elements],
        }

    @staticmethod
    def collate_fn_coreml(elements):
        return {
            "image": [element["image"] for element in elements],
            "anchor_hm": torch.stack(
                [element["anchor_hm"] for element in elements], dim=0
            ),
            "part_hm": torch.stack([element["part_hm"] for element in elements], dim=0),
            "anchor_offsets": torch.stack(
                [element["anchor_offsets"] for element in elements], dim=0
            ),
            "part_offsets": torch.stack(
                [element["part_offsets"] for element in elements], dim=0
            ),
            "embeddings": torch.stack(
                [element["embeddings"] for element in elements], dim=0
            ),
            "anchor_inds": torch.stack(
                [element["anchor_inds"] for element in elements], dim=0
            ),
            "part_inds": torch.stack(
                [element["part_inds"] for element in elements], dim=0
            ),
            "anchor_mask": torch.stack(
                [element["anchor_mask"] for element in elements], dim=0
            ),
            "part_mask": torch.stack(
                [element["part_mask"] for element in elements], dim=0
            ),
            "annotation": [element["annotation"] for element in elements],
        }

    def __repr__(self):
        description = f"Images: {len(self)}\n"
        stats = DatasetStats()

        for file in self.files:
            annotation = ImageAnnotation.from_json(file, self.args.anchor_name)
            stats.update(annotation.objects)

        description += f"{stats}"

        return description

    def histogram(self):
        import altair as alt
        import pandas as pd

        data = []
        for file in self.files:
            annotation = ImageAnnotation.from_json(file, self.args.anchor_name)
            for obj in annotation.objects:
                data.append({"label": obj.name, "leaves": obj.nb_parts})

        df = pd.DataFrame(data)
        mf = df[df.label == "maize"]
        bf = df[df.label == "bean"]

        maize_chart = (
            alt.Chart(mf, width=600, height=400)
            .mark_rect()
            .encode(
                alt.X("leaves:O", title="Nombre de feuilles"),
                alt.Y("count()", title="Nombre de plantes"),
            )
        )

        bean_chart = (
            alt.Chart(bf, width=600, height=400)
            .mark_rect()
            .encode(
                alt.X("leaves:O", title="Nombre de feuilles"),
                alt.Y("count()", title="Nombre de plantes"),
            )
        )

        (maize_chart | bean_chart).show()


class PredictionDataset(data.Dataset):
    def __init__(self, directory, transform=None):
        self.images = sorted(files_with_extension(directory, ".jpg"))
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        img = Image.open(image)
        img_w, img_h = img.size

        if self.transform is not None:
            img = self.transform(img)

        return {"img": img, "img_size": (img_w, img_h)}

    def __len__(self):
        return len(self.images)


class LabelStats:
    def __init__(self):
        self.count = 0
        self.parts = defaultdict(int)

    def __len__(self):
        return len(self.parts)

    def update(self, obj):
        self.count += 1

        for kp in obj.parts:
            self.parts[kp.kind] += 1

    def __repr__(self):
        parts = "{"
        for name, count in self.parts.items():
            parts += f"'{name}': {count}"
        parts += "}"

        description = f"  count: {self.count}\n"
        description += f"  part count: {parts}\n"
        return description


class DatasetStats:
    def __init__(self):
        self.stats = defaultdict(LabelStats)

    def __getitem__(self, label):
        return self.stats[label]

    def __len__(self):
        return len(self.stats)

    def items(self):
        return self.stats.items()

    def update(self, objects):
        if isinstance(objects, list):
            for obj in objects:
                self.stats[obj.name].update(obj)
        else:
            self.stats[objects.name].update(objects)

    def __repr__(self):
        description = ""
        for label, stats in self.items():
            description += f"label: {label}\n"
            description += f"{stats}"
        return description

from ..utils import *
import torch
import torch.utils.data as data
from PIL import Image, ImageOps
from .transforms import Compose


class TrainDataset(data.Dataset):

    def __init__(self, directories, transforms=None):
        super().__init__()

        if isinstance(transforms, list):
            self.transform = Compose(transforms)
        else:
            self.transform = transforms

        if isinstance(directories, (str, Path)):
            self.files = files_with_extension(directories, ".json")
        elif isinstance(directories, list):
            self.files = [file
                for directory in directories
                for file in files_with_extension(directory, ".json")]
        else:
            raise ValueError("'directories' should be either a unique directory or a list of directories.")

        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        annotation = TreeAnnotation.from_json_ann(self.files[index])
        image = Image.open(annotation.image_path)
        image = ImageOps.exif_transpose(image)

        if self.transform is not None:
            encoded = self.transform(image, annotation)
            del encoded["annotation"]
            return encoded
        return image, annotation


class ValidDataset(data.Dataset):

    def __init__(self, directories, transforms=None):
        super().__init__()

        if isinstance(transforms, list):
            self.transform = Compose(transforms)
        else:
            self.transform = transforms

        if isinstance(directories, (str, Path)):
            self.files = files_with_extension(directories, ".json")
        elif isinstance(directories, list):
            self.files = [file
                for directory in directories
                for file in files_with_extension(directory, ".json")]
        else:
            raise ValueError("'directories' should be either a unique directory or a list of directories.")

        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        annotation = TreeAnnotation.from_json_ann(self.files[index])
        image = Image.open(annotation.image_path)
        image = ImageOps.exif_transpose(image)

        if self.transform is not None:
            return self.transform(image, annotation)
        return image, annotation

    def localize_image_names(self):
        for file in self.files:
            annotation = json.loads(file.read_text())
            image_name = Path(annotation["image_path"]).name
            annotation["image_path"] = file.parent / image_name
            file.write_text(json.dumps(annotation, indent=2))

    @staticmethod
    def collate_fn(elements: list[dict]):
        return {
            "image": torch.stack([element["image"] for element in elements], dim=0),
            "annotation": [element["annotation"] for element in elements],
            "heatmaps": torch.stack([element["heatmaps"] for element in elements], dim=0),
            "offsets": torch.stack([element["offsets"] for element in elements], dim=0),
            "embeddings": torch.stack([element["embeddings"] for element in elements], dim=0),
            "inds": torch.stack([element["inds"] for element in elements], dim=0),
            "off_mask": torch.stack([element["off_mask"] for element in elements], dim=0),
            "emb_mask": torch.stack([element["emb_mask"] for element in elements], dim=0)}


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
from ..utils import *
import torch
import torch.utils.data as data
from PIL import Image
from .transforms import Compose


class Dataset(data.Dataset):

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
            self.files = [file
                for directory in directories
                for file in files_with_extension(directory, ".json")]
        else:
            raise ValueError("'directories' should be either a unique directory or a list of directories.")

        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        annotation = GraphAnnotation.from_json_ann(self.files[index])
        image = Image.open(annotation.image_path)

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
    def collate_fn(elements):
        return {
            "image": torch.stack([element["image"] for element in elements], dim=0),
            "anchor_hm": torch.stack([element["anchor_hm"] for element in elements], dim=0),
            "part_hm": torch.stack([element["part_hm"] for element in elements], dim=0),
            "anchor_offsets": torch.stack([element["anchor_offsets"] for element in elements], dim=0),
            "part_offsets": torch.stack([element["part_offsets"] for element in elements], dim=0),
            "embeddings": torch.stack([element["embeddings"] for element in elements], dim=0),
            "anchor_inds": torch.stack([element["anchor_inds"] for element in elements], dim=0),
            "part_inds": torch.stack([element["part_inds"] for element in elements], dim=0),
            "anchor_mask": torch.stack([element["anchor_mask"] for element in elements], dim=0),
            "part_mask": torch.stack([element["part_mask"] for element in elements], dim=0),
            "annotation": [element["annotation"] for element in elements]
        }


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
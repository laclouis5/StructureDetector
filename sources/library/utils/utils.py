import os
import copy
import json
import random
import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict


class Keypoint:

    def __init__(self, kind, x, y, score=None):
        self.kind = kind
        self.x = x
        self.y = y
        self.score = score

    def resize(self, in_size, out_size):
        (img_w, img_h) = in_size
        (new_w, new_h) = out_size

        self.x = self.x / img_w * new_w
        self.y = self.y / img_h * new_h

        return self    

    def resized(self, in_size, out_size):
        return copy.deepcopy(self).resize(in_size, out_size)

    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def normalize(self, size):
        self.x = self.x / size[0]
        self.y = self.y / size[1]
        return self

    def normalized(self, size):
        return copy.deepcopy(self).normalize(size)

    def json_repr(self):
        return {"kind": self.kind, "location": {"x": self.x, "y": self.y}, "score": self.score}

    @staticmethod
    def from_json(json_dict):
        kind = json_dict["kind"]
        location = json_dict["location"]
        (x, y) = location["x"], location["y"]
        score = json_dict.get("score")

        return Keypoint(kind, x, y, score)

    def __repr__(self):
        return f"Keypoint(kind: {self.kind}, x: {self.x}, y: {self.y}, score: {self.score})"


class Box:

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    @property
    def x_mid(self):
        return (self.x_max + self.x_min) / 2
    
    @property
    def y_mid(self):
        return (self.y_max + self.y_min) / 2

    @property
    def width(self):
        return abs(self.x_max - self.x_min)
    
    @property
    def height(self):
        return abs(self.y_max - self.y_min)

    def resize(self, in_size, out_size):
        self.x_min = self.x_min / in_size[0] * out_size[0]
        self.y_min = self.y_min / in_size[1] * out_size[1]
        self.x_max = self.x_max / in_size[0] * out_size[0]
        self.y_max = self.y_max / in_size[1] * out_size[1]
        return self

    def resized(self, in_size, out_size):
        return copy.deepcopy(self).reize(in_size, out_size)

    def normalize(self, size):
        self.x_min = self.x_min / size[0]
        self.y_min = self.y_min / size[1]
        self.x_max = self.x_max / size[0]
        self.y_max = self.y_max / size[1]
        return self

    def normalized(self, size):
        return copy.deepcopy(self).normalize(size)

    def yolo_coords(self, size):
        return (self.x_mid / size[0], self.y_mid / size[1], self.width / size[0], self.height / size[1])

    def standardize(self):
        if self.x_min > self.x_max:
            self.x_min, self.x_max = self.x_max, self.x_min
        if self.y_min > self.y_max:
            self.y_min, self.y_max = self.y_max, self.y_min
        return self

    def standardized(self):
        return copy.deepcopy(self).standardize()

    def json_repr(self):
        return {"x_min": self.x_min, "y_min": self.y_min, "x_max": self.x_max, "y_max": self.y_max}

    @staticmethod
    def from_json(json_dict):
        if json_dict is None:
            return None 
        x_min, y_min, x_max, y_max = json_dict["x_min"], json_dict["y_min"], json_dict["x_max"], json_dict["y_max"]
        return Box(x_min, y_min, x_max, y_max)

    def __repr__(self):
        return f"Box(x_min: {self.x_min}, y_min: {self.y_min}, x_max: {self.x_max}, y_max: {self.y_max})"


class Object:

    def __init__(self, name, anchor, parts=None, box=None):
        self.name = name
        self.anchor = anchor
        self.parts = parts if parts else []
        self.box = box

    @property
    def x(self):
        return self.anchor.x

    @property
    def y(self):
        return self.anchor.y

    @x.setter
    def x(self, new_value):
        self.anchor.x = new_value

    @y.setter
    def y(self, new_value):
        self.anchor.y = new_value

    def resize(self, in_size, out_size):
        self.anchor.resize(in_size, out_size)

        if self.box:
            self.box.resize(in_size, out_size)

        for part in self.parts:
            part.resize(in_size, out_size)

        return self

    def resized(self, in_size, out_size):
        return copy.deepcopy(self).resize(in_size, out_size)

    def distance(self, other):
        return self.anchor.distance(other.anchor)

    def normalize(self, size):
        self.anchor.normalize(size)
        if self.box: 
            self.box.normalize(size)

        for part in self.parts:
            part.normalize(size)

        return self

    def normalized(self, size):
        return copy.deepcopy(self).normalize(size)

    def json_repr(self):
        parts = [self.anchor.json_repr()]
        parts += (part.json_repr() for part in self.parts)
        box = self.box.json_repr() if self.box else None
        return {"label": self.name, "box": box, "parts": parts}

    @staticmethod
    def from_json(json_dict, anchor_name):
        name = json_dict["label"]
        box = Box.from_json(json_dict["box"])
        anchor = None
        parts = []

        for part in json_dict["parts"]:
            part = Keypoint.from_json(part)
            if part.kind == anchor_name:
                assert anchor is None, f"More than one anchor found for object, achor must be unique."
                anchor = part
            else:
                parts.append(part)

        assert anchor is not None, f"Anchor part with name '{anchor_name}' not found while decoding JSON file."
        return Object(name, anchor, parts, box)

    @property
    def nb_parts(self):
        return len(self.parts)

    def __repr__(self):
        return f"Object(name: {self.name}, anchor: {self.anchor}, parts: {self.parts}, box: {self.box})"


class ImageAnnotation:

    def __init__(self, image_path, objects=None, img_size=None):
        self.image_path = image_path
        self.objects = objects if objects else []
        self.img_size = img_size

    @property
    def image_name(self):
        return os.path.basename(self.image_path)

    def normalize(self, size=None):
        size = size if size else self.img_size
        assert size, f"Annotation for '{self.image_path}' does not have a size."

        for obj in self.objects:
            obj.normalize(size)
            
        return self
        
    def normalized(self, size=None):
        return copy.deepcopy(self).normalize(size)

    @staticmethod
    def from_json(file, anchor_name):
        with open(file, "r") as f:
            data = json.load(f)
            image_path = data["image_path"]
            img_size = data.get("img_size", None)
            objects = [Object.from_json(obj, anchor_name) for obj in data["objects"]]

            return ImageAnnotation(image_path, objects, img_size)

    def json_repr(self):
        objects = [obj.json_repr() for obj in self.objects]
        repr = {"image_path": self.image_path, "img_size": self.img_size, "objects": objects}
        return repr

    def save_json(self, save_dir=None):
        save_dir = save_dir if save_dir else "detections/"
        mkdir_if_needed(save_dir)
        save_name = os.path.splitext(self.image_name)[0] + ".json"
        save_name = os.path.join(save_dir, save_name)

        repr = self.json_repr()
        data = json.dumps(repr, indent=2)
        with open(save_name, "w") as f:
            f.write(data)

    def __repr__(self):
        name = os.path.basename(self.image_path)
        return f"ImageAnnotation(name: {name}, objects: {self.objects}, img_size: {self.img_size})"

    def __len__(self):
        return len(self.objects)

    @property
    def nb_parts(self):
        count = 0
        for obj in self.objects:
            count += obj.nb_parts
        return count

    def resize(self, in_size, out_size):
        for obj in self.objects:
            obj.resize(in_size, out_size)
        return self

    def resized(self, in_size, out_size):
        return copy.deepcopy(self).resize(in_size, out_size)

    @property
    def is_empty(self):
        return len(self) == 0


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value):
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count

        return self.avg


def files_with_extension(folder, extension):
    return [os.path.join(folder, file)
        for file in os.listdir(folder)
        if os.path.splitext(file)[1] == extension]


def change_extension(file, extension):
    return os.path.splitext(file)[0] + extension


def change_path(file, path):
    return os.path.join(path, os.path.basename(file))


def mkdir_if_needed(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def save(content, file_name):
    mkdir_if_needed(os.path.split(file_name)[0])

    with open(file_name, "w") as f:
        f.write(content)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# feat: (B, J, C), ind: (B, N)
def gather(feat, ind):
    ind = ind.unsqueeze(-1).expand(-1, -1, feat.size(2))  #  (B, N, C)
    feat = feat.gather(1, ind) # (B, N, C)
    return feat  # (B, N, C)


# feat: (B, C, H, W), ind: (B, N)
def transpose_and_gather(feat, ind):
    ind = ind.unsqueeze(1).expand(-1, feat.size(1), -1)  # (B, C, N)
    feat = feat.view(feat.size(0), feat.size(1), -1)  # (B, C, J = H * W)
    feat = feat.gather(2, ind)  # (B, C, N)
    feat = feat.permute(0, 2, 1)  # (B, N, C)
    return feat  # (B, N, C)


# input: Tensor
def clamped_sigmoid(input):
    output = torch.sigmoid(input)
    output = clamp_in_0_1(output)
    return output


def clamp_in_0_1(tensor):
    return torch.clamp(tensor, min=1e-6, max=1-1e-6)


def clip_annotation(annotation, img_size):
    (img_w, img_h) = img_size

    for obj in annotation.objects:
        obj.x = np.clip(obj.x, 0, img_w - 1)
        obj.y = np.clip(obj.y, 0, img_h - 1)

        for part in obj.parts:
            part.x = np.clip(part.x, 0, img_w - 1)
            part.y = np.clip(part.y, 0, img_h - 1)

        if obj.box:
            obj.box.x_min = np.clip(obj.box.x_min, 0, img_w - 1)
            obj.box.x_max = np.clip(obj.box.x_max, 0, img_w - 1)
            obj.box.y_min = np.clip(obj.box.y_min, 0, img_h - 1)
            obj.box.y_max = np.clip(obj.box.y_max, 0, img_h - 1)

    return annotation


def hflip_annotation(annotation, img_size):
    (img_w, _) = img_size

    for obj in annotation.objects:
        obj.x = img_w - obj.x - 1

        for part in obj.parts:
            part.x = img_w - part.x - 1

        if obj.box:
            x_max = img_w - obj.box.x_min - 1
            x_min = img_w - obj.box.x_max - 1
            obj.box.x_min, obj.box.x_max = x_min, x_max

    return annotation


def vflip_annotation(annotation, img_size):
    (_, img_h) = img_size

    for obj in annotation.objects:
        obj.y = img_h - obj.y - 1

        for part in obj.parts:
            part.y = img_h - part.y - 1

        if obj.box:
            y_max = img_h - obj.box.y_min - 1
            y_min = img_h - obj.box.y_max - 1
            obj.box.y_min, obj.box.y_max = y_min, y_max

    return annotation


def resize_annotation(annotation, img_size, new_size):
    (img_w, img_h) = img_size
    (new_w, new_h) = new_size

    for obj in annotation.objects:
        obj.x = obj.x / img_w * new_w
        obj.y = obj.y / img_h * new_h

        for part in obj.parts:
            part.x = part.x / img_w * new_w
            part.y = part.y / img_h * new_h

        if obj.box:
            obj.box.x_min = obj.box.x_min / img_w * new_w
            obj.box.x_max = obj.box.x_max / img_w * new_w
            obj.box.y_min = obj.box.y_min / img_h * new_h
            obj.box.y_max = obj.box.y_max / img_h * new_h

    return annotation


def gaussian_2d(mu1, mu2, sigma):
    return lambda x, y: np.exp((-(x - mu1)**2 - (y - mu2)**2) / (2 * sigma**2))


# heatmaps: (B, C, H, W)
def nms(heatmaps):
    max_values = nn.functional.max_pool2d(heatmaps, kernel_size=5, stride=1, padding=2)
    return (heatmaps == max_values) * heatmaps  # (B, C, H, W)


# scores: (B, C, H, W)
def topk(scores, k=100):
    (batch, cat, _, width) = scores.size()

    # (B, C, K)
    (topk_scores, topk_inds) = torch.topk(scores.view(batch, cat, -1), k)

    # (B, C, K)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # (B, K)
    (topk_score, topk_ind) = torch.topk(topk_scores.view(batch, -1), k)
    topk_clses = (topk_ind // k)

    # (B, K)
    topk_inds = gather(topk_inds.view(batch, -1, 1), topk_ind).squeeze(-1)
    topk_ys = gather(topk_ys.view(batch, -1, 1), topk_ind).squeeze(-1)
    topk_xs = gather(topk_xs.view(batch, -1, 1), topk_ind).squeeze(-1)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs  # (B, K)


def dict_grouping(iterable, key):
    output = defaultdict(list)
    for element in iterable:
        output[key(element)].append(element)
    return output

import copy
import json
from typing import Callable, Hashable, Sequence, TypeVar
import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
import hashlib


Size = tuple[int, int]

class Keypoint:

    def __init__(self, kind: str, x: float, y: float, score: float = None):
        self.kind = kind
        self.x = x
        self.y = y
        self.score = score

    def resize(self, in_size: Size, out_size: Size) -> "Keypoint":
        img_w, img_h = in_size
        new_w, new_h = out_size

        self.x *= new_w / img_w
        self.y *= new_h / img_h

        return self

    def resized(self, in_size: Size, out_size: Size) -> "Keypoint":
        return copy.deepcopy(self).resize(in_size, out_size)

    def distance(self, other: "Keypoint") -> float:
        return np.hypot(self.x - other.x, self.y - other.y)

    def normalize(self, size: Size) -> "Keypoint":
        w, h = size
        self.x /= w
        self.y /= h
        return self

    def normalized(self, size: Size) -> "Keypoint":
        return copy.deepcopy(self).normalize(size)

    def clip(self, size: Size) -> "Keypoint":
        x_max, y_max = size
        self.x = min(max(self.x, 0), x_max)
        self.y = min(max(self.y, 0), y_max)
        return self

    def clipped(self, size: Size) -> "Keypoint":
        return copy.deepcopy(self).clip(size)

    def json_repr(self) -> dict:
        return {"kind": self.kind, "location": {"x": self.x, "y": self.y}, "score": self.score}

    @staticmethod
    def from_json(json_dict: dict) -> "Keypoint":
        kind = json_dict["kind"]
        location = json_dict["location"]
        (x, y) = location["x"], location["y"]
        score = json_dict.get("score")

        return Keypoint(kind, x, y, score)

    def __repr__(self) -> str:
        return f"Keypoint(kind: {self.kind}, x: {self.x}, y: {self.y}, score: {self.score})"


class Graph:

    def __init__(self, keypoints: Sequence[Keypoint] = None):
        if keypoints is not None:
            self._adjacency = {kp: set() for kp in keypoints}
        else:
            self._adjacency = {}

    @property
    def keypoints(self):
        return self._adjacency.keys()

    @property
    def nb_keypoints(self) -> int:
        return len(self._adjacency)

    def neighbors(self, keypoint: Keypoint) -> "set[Keypoint]":
        assert keypoint in self.keypoints
        return self._adjacency[keypoint]

    def connect(self, kp1: Keypoint, kp2: Keypoint):
        assert kp1 in self.keypoints
        assert kp2 in self.keypoints
        
        self._adjacency[kp1].add(kp2)
        self._adjacency[kp2].add(kp1)

    def add(self, keypoint: Keypoint):
        assert keypoint not in self.keypoints
        self._adjacency[keypoint] = set()

    def connected_graphs(self) -> "list[Graph]":
        connected_components = []
        visited = set()

        def _gather_connected(keypoint: Keypoint, connected: "list[Keypoint]"):
            visited.add(keypoint)
            connected.append(keypoint)

            for neighbor in self.neighbors(keypoint):
                if neighbor not in visited:
                    _gather_connected(neighbor, connected)

        for keypoint in self.keypoints:
            if keypoint not in visited:
                connected = []
                _gather_connected(keypoint, connected)
                connected_components.append(connected)

        graphs = [Graph(connected) for connected in connected_components]

        for graph in graphs:
            for keypoint in graph.keypoints:
                for neighbor in self.neighbors(keypoint):
                    graph.connect(keypoint, neighbor)
        
        return graphs

    def resize(self, in_size: Size, out_size: Size) -> "Graph":
        for keypoint in self.keypoints:
            keypoint.resize(in_size, out_size)
        return self

    def resized(self, in_size: Size, out_size: Size) -> "Graph":
        return copy.deepcopy(self).resize(in_size, out_size)

    def normalize(self, size: Size) -> "Graph":
        for keypoint in self.keypoints:
            keypoint.normalize(size)
        return self

    def normalized(self, size: Size) -> "Graph":
        return copy.deepcopy(self).normalize(size)

    def clip(self, size: Size) -> "Graph":
        for keypoint in self.keypoints:
            keypoint.clip(size)
        return self

    def clipped(self, size: Size) -> "Graph":
        return copy.deepcopy(self).clip(size)

    def json_repr(self) -> dict:
        inds = {kp: i for i, kp in enumerate(self.keypoints)}

        return {
            "keypoints": [kp.json_repr() for kp in self.keypoints],
            "adjacency": [[inds[neighbor] for neighbor in neighbors] 
                for neighbors in self._adjacency.values()]}

    @staticmethod
    def from_json_ann(json_obj: dict) -> "Graph":
        graph = Graph()

        def _fill_graph(node: dict):
            value = node["value"]
            x, y = value["x"], value["y"]
            keypoint = Keypoint(kind=value.get("name", "unknown"), x=x, y=y)
            node["_kp"] = keypoint
            graph.add(keypoint)

            for child in node["children"]:
                _fill_graph(child)
                graph.connect(child["_kp"], keypoint)

        _fill_graph(json_obj)

        return graph

    @staticmethod
    def from_json(json_obj: dict) -> "Graph":
        keypoints = [Keypoint.from_json(kp) for kp in json_obj["keypoints"]]
        graph = Graph(keypoints)
        
        for i, adjacency in enumerate(json_obj["adjacency"]):
            for j in adjacency:
                graph.connect(keypoints[i], keypoints[j])

        return graph

    def __repr__(self) -> str:
        return f"Graph(keypoints: {set(self.keypoints)})"


class GraphAnnotation:

    def __init__(self, image_path: Path, graph: Graph, image_size: "tuple[int, int]" = None):
        self.image_path = image_path
        self.graph = graph
        self.image_size = image_size

    @property
    def image_name(self) -> str:
        return self.image_path.name
    
    def resize(self, in_size: Size, out_size: Size) -> "GraphAnnotation":
        self.graph.resize(in_size, out_size)
        return self

    def resized(self, in_size: Size, out_size: Size) -> "GraphAnnotation":
        return copy.deepcopy(self).resize(in_size, out_size)

    def flip_lr(self, size: Size) -> "GraphAnnotation":
        img_w, _ = size
        for keypoint in self.graph.keypoints:
            keypoint.x = img_w - keypoint.x - 1
        return self

    def fliped_lr(self, size: Size) -> "GraphAnnotation":
        return copy.deepcopy(self).flip_lr(size)

    def clip(self, size: Size) -> "GraphAnnotation":
        self.graph.clip(size)
        return self

    def clipped(self, size: Size) -> "GraphAnnotation":
        return copy.deepcopy(self).clip(size)

    def json_repr(self) -> dict:
        image_size = self.image_size
        if image_size is not None:
            img_w, img_h = self.image_size
            image_size = {"width": img_w, "height": img_h}

        return {
            "image_path": f"{self.image_path}",
            "image_size": image_size,
            "graph": self.graph.json_repr()}

    @staticmethod
    def _from_json_obj(json_obj: dict) -> "GraphAnnotation":
        img_size = json_obj.get("image_size")  
        if img_size is not None:
            img_size = (img_size["width"], img_size["height"])

        return GraphAnnotation(
            image_path=Path(json_obj["image_path"]), 
            graph=Graph.from_json(json_obj["graph"]),
            image_size=img_size)

    @staticmethod
    def _from_json_ann_obj(json_obj: dict) -> "GraphAnnotation":
        img_size = json_obj.get("imageSize")  
        if img_size is not None:
            img_size = (img_size["width"], img_size["height"])

        return GraphAnnotation(
            image_path=Path(json_obj["imageUrl"]), 
            graph=Graph.from_json_ann(json_obj["tree"]),
            image_size=img_size)

    @staticmethod
    def from_json(file_path: Path) -> "GraphAnnotation":
        content = json.loads(file_path.read_text())
        return GraphAnnotation._from_json_obj(content)

    @staticmethod
    def from_json_ann(file_path: Path) -> "GraphAnnotation":
        content = json.loads(file_path.read_text())
        return GraphAnnotation._from_json_ann_obj(content)


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


def files_with_extension(folder: Path, extension: str, recursive: bool = False) -> Sequence[Path]:
    extension = extension if extension.startswith(".") else f".{extension}"
    return folder.rglob(f"*{extension}") if recursive else folder.glob(f"*{extension}")


def mkdir_if_needed(directory: Path):
    Path(directory).mkdir(exist_ok=True)


def set_seed(seed="1975846251"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# feat: (B, J, C), ind: (B, N)
def gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    ind = ind.unsqueeze(-1).expand(-1, -1, feat.size(2))  #  (B, N, C)
    return feat.gather(1, ind) # (B, N, C)


# feat: (B, C, H, W), ind: (B, N)
def transpose_and_gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    ind = ind.unsqueeze(1).expand(-1, feat.size(1), -1)  # (B, C, N)
    feat = feat.view(feat.size(0), feat.size(1), -1)  # (B, C, J = H * W)
    feat = feat.gather(2, ind)  # (B, C, N)
    return feat.permute(0, 2, 1)  # (B, N, C)


# input: Tensor
def clamped_sigmoid(input: torch.Tensor) -> torch.Tensor:
    output = torch.sigmoid(input)
    return clamp_in_0_1(output)


def clamp_in_0_1(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, min=1e-6, max=1-1e-6)


def gaussian_2d(X: np.ndarray, Y: np.ndarray, mu1: float, mu2: float, sigma: float) -> np.ndarray:
    return torch.exp((-(X - mu1)**2 - (Y - mu2)**2) / (2 * sigma**2))


# heatmaps: (B, C, H, W)
def nms(heatmaps: torch.Tensor) -> torch.Tensor:
    max_values = nn.functional.max_pool2d(heatmaps, kernel_size=5, stride=1, padding=2)
    return (heatmaps == max_values) * heatmaps  # (B, C, H, W)


# scores: (B, C, H, W)
def topk(
    scores: torch.Tensor, k: int = 100
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, cat, _, width = scores.size()

    # (B, C, K)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)

    # (B, C, K)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # (B, K)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
    topk_clses = (topk_ind // k)

    # (B, K)
    topk_inds = gather(topk_inds.view(batch, -1, 1), topk_ind).squeeze(-1)
    topk_ys = gather(topk_ys.view(batch, -1, 1), topk_ind).squeeze(-1)
    topk_xs = gather(topk_xs.view(batch, -1, 1), topk_ind).squeeze(-1)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs  # (B, K)


T = TypeVar("T")
V = TypeVar("V", bound=Hashable)
def dict_grouping(iterable: Sequence[T], key: Callable[[T], V]) -> dict[V, T]:
    output = defaultdict(list)
    for element in iterable:
        output[key(element)].append(element)
    return output


Color = tuple[int, int, int]
def unique_color(string: str, seed: str = "") -> Color:
    return (*hashlib.md5((string + seed).encode()).digest()[:3],)


def get_unique_color_map(labels: Sequence[str]) -> dict[str, Color]:
    """(╯°□°)╯︵ ┻━hash()━┻"""
    return {n: unique_color(n) for n in labels}
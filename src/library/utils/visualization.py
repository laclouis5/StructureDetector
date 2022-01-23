from .utils import *
from PIL import ImageDraw, ImageOps, Image
from PIL.Image import Image as PILImage
import torchvision.transforms.functional as F
import torch


def un_normalize(tensor: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W)
    # (3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device)[..., None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device)[..., None, None]
    return tensor * std + mean


def draw_graph(image: PILImage, graph: Graph) -> PILImage:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    img_size = image.size
    radius = int(min(img_size) * 0.5/100)
    thickness = int(radius / 2)

    for keypoint in graph.keypoints:
        x1, y1 = keypoint.x, keypoint.y

        neighbors = graph.neighbors(keypoint)
        for neighbor in neighbors:
            x2, y2 = neighbor.x, neighbor.y

            draw.line([x1, y1, x2, y2], fill="white", width=thickness)

    for keypoint in graph.keypoints:
        x, y = keypoint.x, keypoint.y
        kp_color = unique_color(keypoint.kind)
        
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
            fill=kp_color, outline=kp_color)

    return image


def draw_tree(image: PILImage, tree: Tree) -> PILImage:
    return draw_graph(image, tree.to_graph())


# anchor_hm: (M, H, W)
def draw_heatmaps(anchor_hm: torch.Tensor, args) -> PILImage:
    assert anchor_hm.dim() == 3, "Do not send batched data to this function, only one sample"

    c, h, w = anchor_hm.shape  # (M, H, W)

    obj_colors = torch.tensor(
        [unique_color(label) for label in args.labels],
        device=anchor_hm.device)  # (M, 3)

    output = torch.zeros(c, 3, h, w)  # (M, 3, H, W)
    for i, hm in enumerate(anchor_hm):  # (H, W)
        color = obj_colors[i]  # (3)
        hm = hm[None, ...] * color[:, None, None]  # (3, H, W)
        output[i, ...] = hm

    output = output.mean(dim=0)  # (3, H, W)
    output = output / output.max() * 255
    return F.to_pil_image(output.type(torch.uint8))  # (3, H, W)


def draw_embeddings(image: PILImage, embeddings: torch.Tensor, args) -> PILImage:
    assert embeddings.shape[0] == 1, "BS should be one"
    image = image.copy()
    draw = ImageDraw.Draw(image)

    embeddings = embeddings[0] * args.down_ratio  # (2, H, W)
    embeddings = embeddings.permute(1, 2, 0)  # (H, W, 2)

    thickness = int(min(image.size) * 0.5/100)

    for y in range(0, embeddings.shape[0], 4):
        for x in range(0, embeddings.shape[1], 4):
            x1 = x * args.down_ratio
            y1 = y * args.down_ratio

            x2 = (embeddings[y, x, 0] + x1).item()
            y2 = (embeddings[y, x, 1] + y1).item()

            draw.line([x1, y1, x2, y2], fill=(255, 0, 0), width=thickness)

    return image
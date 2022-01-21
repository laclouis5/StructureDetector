from turtle import width
from .utils import *
from PIL import ImageDraw
from PIL.Image import Image as PILImage
import torchvision.transforms.functional as F
import torch


def un_normalize(tensor: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W)
    # (3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device)[..., None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device)[..., None, None]
    return tensor * std + mean


def draw_graph(image: PILImage, annotation: GraphAnnotation, args) -> PILImage:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    graph = annotation.graph

    img_size = image.size
    radius = int(min(img_size) * 2/100)
    thickness = radius / 2

    for keypoint in graph.keypoints:
        x1, y1 = keypoint.x, keypoint.y
        kp_color = args._label_color_map[keypoint.kind]

        neighbors = graph.neighbors(keypoint)
        for neighbor in neighbors:
            x2, y2 = neighbor.x, neighbor.y

            draw.line([x1, y1, x2, y2], fill="white", width=thickness)

        draw.ellipse([x1 - radius, y1 - radius, x1 + radius, y1 + radius],
            fill=kp_color, outline=kp_color)

    return image


def draw_heatmaps(anchor_hm: torch.Tensor, part_hm: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    assert anchor_hm.dim() == 3 and part_hm.dim() == 3, "Do not send batched data to this function, only one sample"

    (c1, h, w) = anchor_hm.shape
    c2 = part_hm.shape[0]

    obj_colors = torch.tensor([args._label_color_map.get(args._r_labels.get(i, None), (0, 0, 0)) for i in range(c1)], device=anchor_hm.device)
    part_colors = torch.tensor([args._part_color_map.get(args._r_parts.get(i, None), (0, 0, 0)) for i in range(c2)], device=part_hm.device)

    obj_colors = obj_colors[..., None, None].expand(-1, -1, h, w)
    part_colors = part_colors[..., None, None].expand(-1, -1, h, w)

    (anchor_hm_max, obj_inds) = torch.max(anchor_hm, 0)
    (part_hm_max, part_inds) = torch.max(part_hm, 0)
    obj_inds = obj_inds[None, None, ...].expand(-1, 3, -1, -1)
    part_inds = part_inds[None, None, ...].expand(-1, 3, -1, -1)

    anchor_hm_color = torch.gather(obj_colors, dim=0, index=obj_inds).squeeze()
    part_hm_color = torch.gather(part_colors, dim=0, index=part_inds).squeeze()

    anchor_hm_color = anchor_hm_color.float() * anchor_hm_max
    part_hm_color = part_hm_color.float() * part_hm_max

    return anchor_hm_color.type(torch.uint8), part_hm_color.type(torch.uint8)


def draw_kp_and_emb(
    image: PILImage, topk_obj: torch.Tensor, topk_kp: torch.Tensor, embeddings: torch.Tensor, args
):
    thresh = args.conf_threshold

    img = un_normalize(image.cpu())
    img = F.to_pil_image(img)  # This converts from [0, 1] to [0, 255]
    draw = ImageDraw.Draw(img)
    (img_w, img_h) = img.size
    offset = int(min(img_w, img_h) * 1/100)
    thickness = int(min(img_w, img_h) * 1/100)

    obj_scores, _, obj_labels, obj_ys, obj_xs = topk_obj
    part_scores, _, part_labels, part_ys, part_xs = topk_kp

    for (x, y, label, score) in zip(obj_xs.squeeze(0), obj_ys.squeeze(0), obj_labels.squeeze(0), obj_scores.squeeze(0)):
        if score < thresh: continue
        color = args._label_color_map[args._r_labels[label.item()]]

        x *= args.down_ratio
        y *= args.down_ratio

        draw.ellipse([x - offset, y - offset, x + offset, y + offset],
            fill=color, outline=color)

    for (x, y, label, score, embeddings) in zip(part_xs.squeeze(0), part_ys.squeeze(0), part_labels.squeeze(0), part_scores.squeeze(0), embeddings.squeeze(0)):
        if score < thresh: continue
        color = args._part_color_map[args._r_parts[label.item()]]

        x *= args.down_ratio
        y *= args.down_ratio

        e_x = x + args.down_ratio * embeddings[0]
        e_y = y + args.down_ratio * embeddings[1]

        draw.ellipse([x - offset, y - offset, x + offset, y + offset],
            fill=color, outline=color)

        draw.line([x, y, e_x, e_y], fill=color, width=thickness)

    return img


def draw_embeddings(image, embeddings, args):
    assert embeddings.shape[0] == 1, "BS should be one"

    embeddings = embeddings[0] * args.down_ratio  # (2, H, W)
    embeddings = embeddings.permute(1, 2, 0)  # (H, W, 2)
    image = F.to_pil_image(un_normalize(image.cpu()))
    draw = ImageDraw.Draw(image)

    thickness = int(min(image.size) * 0.5/100)

    for y in range(0, embeddings.shape[0], 4):
        for x in range(0, embeddings.shape[1], 4):
            x1 = x * args.down_ratio
            y1 = y * args.down_ratio

            x2 = (embeddings[y, x, 0] + x1).item()
            y2 = (embeddings[y, x, 1] + y1).item()

            draw.line([x1, y1, x2, y2], fill=(255, 0, 0), width=thickness)

    return image
    

def draw_keypoints(image: PILImage, keypoints: Sequence[Keypoint], args) -> PILImage:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    img_w, img_h = image.size
    offset = int(min(img_w, img_h) * 1/100)

    for kp in keypoints:
        if kp.kind in args.labels.keys():
            color = args._label_color_map[kp.kind]
        elif kp.kind in args.parts.keys():
            color = args._part_color_map[kp.kind]
        else: raise ValueError

        draw.ellipse([kp.x - offset, kp.y - offset, kp.x + offset, kp.y + offset],
            fill=color, 
            outline=color)

    return image
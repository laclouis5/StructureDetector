from .utils import *
from PIL import Image, ImageDraw
import numpy.random as random
import torchvision.transforms.functional as F
import torch


def get_random_color_map(count, seed=None):
    if seed: random.seed(seed)
    return random.randint(0, 256, (count, 3))


def un_normalize(tensor):  # (B, 3, H, W)
    # (3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).unsqueeze(-1).unsqueeze(-1)
    return tensor * std + mean


def draw(image, annotation, args, unnorm_image=True):
    if isinstance(image, torch.Tensor):
        img = un_normalize(image) if unnorm_image else image
        img = F.to_pil_image(img.cpu())  # This converts from [0, 1] to [0, 255]
    else:
        img = image.copy()

    draw = ImageDraw.Draw(img)
    (img_w, img_h) = img.size
    offset = int(min(img_w, img_h) * 1/100)
    thickness = int(min(img_w, img_h) * 1/100)

    object_cmap = get_random_color_map(len(args.labels), 7567)
    part_cmap = get_random_color_map(len(args.parts), 9456)

    object_cmap = {label: (*color,) for (label, color) in zip(args.labels.keys(), object_cmap)}
    part_cmap = {part: (*color,) for (part, color) in zip(args.parts.keys(), part_cmap)}

    for obj in annotation.objects:
        obj_color = object_cmap[obj.name]

        (x, y) = (obj.x, obj.y)

        for kp in obj.parts:
            kp_color = part_cmap[kp.kind]

            draw.ellipse([kp.x - offset, kp.y - offset, kp.x + offset, kp.y + offset],
                fill=kp_color, outline=kp_color)
            draw.line([x, y, kp.x, kp.y],
                fill=kp_color, width=thickness)

        draw.ellipse([x - offset, y - offset, x + offset, y + offset],
            fill=obj_color, outline=obj_color)

        # if obj.box:
        #     box = obj.box
        #     draw.rectangle([box.x_min, box.y_min, box.x_max, box.y_max], outline=obj_color, width=thickness)

    return img


def draw_heatmaps(anchor_hm, part_hm, args):
    assert anchor_hm.dim() == 3 and part_hm.dim() == 3, "Do not send batched data to this function, only one sample"

    (h, w) = anchor_hm.shape[1:]

    object_cmap = get_random_color_map(len(args.labels), 7567)
    part_cmap = get_random_color_map(len(args.parts), 9456)

    obj_colors = torch.from_numpy(object_cmap).to(anchor_hm.device)
    part_colors = torch.from_numpy(part_cmap).to(anchor_hm.device)
    obj_colors = obj_colors.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
    part_colors = part_colors.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

    (anchor_hm_max, obj_inds) = torch.max(anchor_hm, 0)
    (part_hm_max, part_inds) = torch.max(part_hm, 0)
    obj_inds = obj_inds.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
    part_inds = part_inds.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

    anchor_hm_color = torch.gather(obj_colors, dim=0, index=obj_inds).squeeze()
    part_hm_color = torch.gather(part_colors, dim=0, index=part_inds).squeeze()

    anchor_hm_color = anchor_hm_color.float() * anchor_hm_max
    part_hm_color = part_hm_color.float() * part_hm_max

    return anchor_hm_color.type(torch.uint8), part_hm_color.type(torch.uint8)


def draw_kp_and_emb(image, topk_obj, topk_kp, embeddings, args):
    thresh = args.conf_threshold

    img = un_normalize(image.cpu())
    img = F.to_pil_image(img)  # This converts from [0, 1] to [0, 255]
    draw = ImageDraw.Draw(img)
    (img_w, img_h) = img.size
    offset = int(min(img_w, img_h) * 1/100)
    thickness = int(min(img_w, img_h) * 1/100)

    object_cmap = get_random_color_map(len(args.labels), 7567)
    part_cmap = get_random_color_map(len(args.parts), 9456)

    (obj_scores, _, obj_labels, obj_ys, obj_xs) = topk_obj
    (part_scores, _, part_labels, part_ys, part_xs) = topk_kp

    for (x, y, label, score) in zip(obj_xs.squeeze(0), obj_ys.squeeze(0), obj_labels.squeeze(0), obj_scores.squeeze(0)):
        if score < thresh: continue
        color = (*object_cmap[label],)

        x *= args.down_ratio
        y *= args.down_ratio

        draw.ellipse([x - offset, y - offset, x + offset, y + offset],
            fill=color, outline=color)

    for (x, y, label, score, embeddings) in zip(part_xs.squeeze(0), part_ys.squeeze(0), part_labels.squeeze(0), part_scores.squeeze(0), embeddings.squeeze(0)):
        if score < thresh: continue
        color = (*part_cmap[label],)

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
    
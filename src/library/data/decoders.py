from ..utils import *
import torch
from collections import defaultdict


class Decoder:

    def __init__(self, args):
        self.label_map = args._r_labels
        self.part_map = args._r_parts
        self.anchor_name = args.anchor_name

        self.args = args
        self.down_ratio = args.down_ratio
        self.max_objects = args.max_objects  # K
        self.max_parts = args.max_parts  # P

    # output: (B, M+N+4, H/R, W/R), see network.py
    def __call__(self, outputs, conf_thresh=None, dist_thresh=None, return_metadata=False):
        conf_thresh = conf_thresh or self.args.conf_threshold
        dist_thresh = dist_thresh or self.args.decoder_dist_thresh

        (out_h, out_w) = outputs["anchor_hm"].shape[2:]  # H/R, W/R
        (in_h, in_w) = int(self.down_ratio * out_h), int(self.down_ratio * out_w)  # H, W

        # Anchors
        anchor_hm_sig = clamped_sigmoid(outputs["anchor_hm"])  # (B, M, H/R, W/R)
        anchor_hm = nms(anchor_hm_sig)  # (B, M, H/R, W/R)
        (anchor_scores, anchor_inds, anchor_labels, anchor_ys, anchor_xs) = topk(
            anchor_hm, k=self.max_objects)  # (B, K)
        anchor_offsets = transpose_and_gather(outputs["offsets"], anchor_inds)  # (B, K, 2)
        anchor_xs += anchor_offsets[..., 0]  # (B, K)
        anchor_ys += anchor_offsets[..., 1]  # (B, K)

        anchor_out = torch.stack((
            anchor_xs, anchor_ys,
            anchor_scores, anchor_labels.float()
        ), dim=2)  # (B, K, 4)

        # Parts
        part_hm_sig = clamped_sigmoid(outputs["part_hm"])  # (B, N, H/R, W/R)
        part_hm = nms(part_hm_sig)  # (B, N, H/R, W/R)
        (part_scores, part_inds, part_labels, part_ys, part_xs) = topk(
            part_hm, k=self.max_parts)  # (B, P)
        part_offsets = transpose_and_gather(outputs["offsets"], part_inds)  # (B, P, 2)
        embeddings = transpose_and_gather(outputs["embeddings"], part_inds)  # (B, P, 2)
        part_xs += part_offsets[..., 0]  # (B, P)
        part_ys += part_offsets[..., 1]  # (B, P)
        origin_xs = part_xs + embeddings[..., 0]  # (B, P)
        origin_ys = part_ys + embeddings[..., 1]  # (B, P)

        part_out = torch.stack((
            part_xs, part_ys,
            part_scores, part_labels.float(), 
            origin_xs, origin_ys
        ), dim=2)  # (B, P, 6)

        # Anchor-part association
        part_mask = (part_scores > conf_thresh).float()  # (B, P)
        part_scores = -(1 - part_mask) + part_mask * part_scores  # (B, P)
        ori_xs = (-1e6*(1 - part_mask) + part_mask * origin_xs)  # (B, P)
        ori_ys = (-1e6*(1 - part_mask) + part_mask * origin_ys)  # (B, P)

        anchor_mask = (anchor_scores > conf_thresh).float()  # (B, K)
        anchor_scores = -(1 - anchor_mask) + anchor_mask * anchor_scores  # (B, K)
        pos_xs = (1e6*(1 - anchor_mask) + anchor_mask * anchor_xs)  # (B, K)
        pos_ys = (1e6*(1 - anchor_mask) + anchor_mask * anchor_ys)  # (B, K)

        anchor_pos = torch.stack((pos_xs, pos_ys), dim=-1)  # (B, K, 2)
        origins = torch.stack((ori_xs, ori_ys), dim=-1)  # (B, P, 2)

        anchor_pos = anchor_pos.unsqueeze(2).expand(-1, -1, self.max_parts, -1)  # (B, K, P, 2)
        origins = origins.unsqueeze(1).expand(-1, self.max_objects, -1, -1)  # (B, K, P, 2)

        sq_distance = torch.hypot(*torch.unbind(origins - anchor_pos, dim=-1))  # (B, K, P)
        (min_vals, min_inds) = sq_distance.min(dim=1)  # (B, P)
        min_vals = min_vals < (dist_thresh * min(out_w, out_h))  # (B, P)

        # Tensor to dynamic array of annotations
        annotations = []
        for b_i, batch in enumerate(min_inds):
            part_list = defaultdict(list)
            image_annotation = ImageAnnotation(f"batch_{b_i}")

            for i, index in enumerate(batch):
                if not min_vals[b_i, i]:
                    continue

                part_list[index.item()].append(part_out[b_i, i])

            for anchor_i, anchor in enumerate(anchor_out[b_i]):
                score = anchor[2].item()
                if score <= conf_thresh: continue

                parts = part_list[anchor_i]
                parts = [Keypoint(kind=self.part_map[int(p[3])], x=p[0].item(), y=p[1].item(), score=p[2].item())
                    for p in parts]
                anchor_label = self.label_map[int(anchor[3])]
                anchor = Keypoint(kind=self.anchor_name, x=anchor[0].item(), y=anchor[1].item(), score=score)
                obj = Object(name=anchor_label, anchor=anchor, parts=parts)
                image_annotation.objects.append(obj)

            annotations.append(image_annotation.resize((out_w, out_h), (in_w, in_h)))

        # Optionally return metadata for debug
        if return_metadata:
            raw_parts = []
            for batch in part_out:
                raw_parts_b = []

                for part in batch:
                    x = part[0].item()
                    y = part[1].item()
                    score = part[2].item()
                    label = self.part_map[int(part[3])]

                    if score < conf_thresh: continue

                    kp = Keypoint(label, x, y, score)
                    raw_parts_b.append(kp.resize((out_w, out_h), (in_w, in_h)))

                raw_parts.append(raw_parts_b)

            return {
                "annotation": annotations, "anchor_hm_sig": anchor_hm_sig, 
                "part_hm_sig": part_hm_sig, "embeddings": embeddings, 
                "topk_anchor": (anchor_scores, anchor_inds, anchor_labels, anchor_ys, anchor_xs), 
                "topk_kp": (part_scores, part_inds, part_labels, part_ys, part_xs), 
                "raw_parts": raw_parts, "raw_embeddings": outputs["embeddings"], "raw_offsets": outputs["offsets"]}
            
        return annotations  # (B)


class KeypointDecoder:

    def __init__(self, args):
        self.label_map = args._r_labels
        self.part_map = args._r_parts
        self.anchor_name = args.anchor_name

        self.args = args
        self.down_ratio = args.down_ratio
        self.max_objects = args.max_objects  # K
        self.max_parts = args.max_parts  # P

    # output: (B, M+N+4, H/R, W/R), see network.py
    def __call__(self, outputs):
        conf_thresh = self.args.conf_threshold
        out_h, out_w = outputs["anchor_hm"].shape[2:]  # H/R, W/R
        in_h, in_w = int(self.down_ratio * out_h), int(self.down_ratio * out_w)  # H, W
        r_h, r_w = in_h / out_h, in_w / out_w

        # Anchors
        anchor_hm_sig = clamped_sigmoid(outputs["anchor_hm"])  # (B, M, H/R, W/R)
        anchor_hm = nms(anchor_hm_sig)  # (B, M, H/R, W/R)
        (anchor_scores, anchor_inds, anchor_labels, anchor_ys, anchor_xs) = topk(
            anchor_hm, k=self.max_objects)  # (B, K)
        anchor_offsets = transpose_and_gather(outputs["offsets"], anchor_inds)  # (B, K, 2)
        anchor_xs += anchor_offsets[..., 0]  # (B, K)
        anchor_ys += anchor_offsets[..., 1]  # (B, K)

        anchors = torch.stack((
            anchor_xs * r_w, anchor_ys * r_h,
            anchor_scores, anchor_labels.float()
        ), dim=2)  # (B, K, 4)

        # Parts
        part_hm_sig = clamped_sigmoid(outputs["part_hm"])  # (B, N, H/R, W/R)
        part_hm = nms(part_hm_sig)  # (B, N, H/R, W/R)
        (part_scores, part_inds, part_labels, part_ys, part_xs) = topk(
            part_hm, k=self.max_parts)  # (B, P)
        part_offsets = transpose_and_gather(outputs["offsets"], part_inds)  # (B, P, 2)
        part_xs += part_offsets[..., 0]  # (B, P)
        part_ys += part_offsets[..., 1]  # (B, P)

        parts = torch.stack((
            part_xs * r_w, part_ys * r_h,
            part_scores, part_labels.float(),
        ), dim=2)  # (B, P, 4)
        
        annotations = []

        for anchor_batch, part_batch in zip(anchors, parts):
            keypoints = []

            for x, y, score, label in anchor_batch:
                if score < conf_thresh: continue
                label = self.label_map[int(label.item())]
                keypoints.append(Keypoint(kind=label, x=x.item(), y=y.item(), score=score.item()))

            for x, y, score, label in part_batch:
                if score < conf_thresh: continue
                label = self.part_map[int(label)]
                keypoints.append(Keypoint(kind=label, x=x.item(), y=y.item(), score=score.item()))
            
            annotations.append(keypoints)

        return annotations
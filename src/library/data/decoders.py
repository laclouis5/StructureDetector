from ..utils import *
import torch


class Decoder:

    def __init__(self, args):
        self.args = args
        self.label_map = args._r_labels
        self.down_ratio = args.down_ratio  # R
        self.max_objects = args.max_objects  # K

    # output: (B, M+4, H/R, W/R)
    def __call__(self, 
        outputs: dict[str, torch.Tensor], conf_thresh: float = None, dist_thresh: float = None
    ) -> list[Graph]:
        conf_thresh: float = conf_thresh if conf_thresh is not None else self.args.conf_threshold
        dist_thresh: float = dist_thresh if dist_thresh is not None else self.args.decoder_dist_thresh

        heatmaps = outputs["heatmaps"]  # (B, M, H/R, W/R)
        offsets = outputs["offsets"]  # (B, 2, H/R, W/R)
        embeddings = outputs["embeddings"]  # (B, 2, H/R, W/R)

        out_h, out_w = heatmaps.shape[2:]  # H/R, W/R
        in_h, in_w = int(self.down_ratio * out_h), int(self.down_ratio * out_w)  # H, W

        # Keypoints
        anchor_hm = nms(heatmaps)  # (B, M, H/R, W/R)
        anchor_scores, anchor_inds, anchor_labels, anchor_ys, anchor_xs = topk(
            anchor_hm, k=self.max_objects)  # (B, K)
        anchor_offsets = transpose_and_gather(offsets, anchor_inds)  # (B, K, 2)
        anchor_xs += anchor_offsets[..., 0]  # (B, K)
        anchor_ys += anchor_offsets[..., 1]  # (B, K)

        anchor_out = torch.stack((
            anchor_xs, anchor_ys,
            anchor_scores, anchor_labels.float()
        ), dim=2)  # (B, K, 4)

        # Embeddings
        # TODO: Check if valid! (anchor_inds, ...)
        embeddings = transpose_and_gather(embeddings, anchor_inds)  # (B, K, 2)
        origin_xs = anchor_xs + embeddings[..., 0]  # (B, K)
        origin_ys = anchor_ys + embeddings[..., 1]  # (B, K)

        # Association
        # TODO: Check if valid (anchor_mask for ori_xs)
        anchor_mask = (anchor_scores > conf_thresh).float()  # (B, K)
        anchor_scores = -(1 - anchor_mask) + anchor_mask * anchor_scores  # (B, K)
        pos_xs = (1e6*(1 - anchor_mask) + anchor_mask * anchor_xs)  # (B, K)
        pos_ys = (1e6*(1 - anchor_mask) + anchor_mask * anchor_ys)  # (B, K)

        ori_xs = (-1e6*(1 - anchor_mask) + anchor_mask * origin_xs)  # (B, K)
        ori_ys = (-1e6*(1 - anchor_mask) + anchor_mask * origin_ys)  # (B, K)

        anchor_pos = torch.stack((pos_xs, pos_ys), dim=-1)  # (B, K, 2)
        origins = torch.stack((ori_xs, ori_ys), dim=-1)  # (B, K, 2)

        anchor_pos = anchor_pos.unsqueeze(2).expand(-1, -1, self.max_objects, -1)  # (B, K, K, 2)
        origins = origins.unsqueeze(1).expand(-1, self.max_objects, -1, -1)  # (B, K, K, 2)

        sq_distance = torch.hypot(*torch.unbind(origins - anchor_pos, dim=-1))  # (B, K, K)
        min_vals, min_inds = sq_distance.min(dim=1)  # (B, K)
        min_vals = min_vals < (dist_thresh * min(out_w, out_h))  # (B, K)

        # Assemble
        graphs = []
        batch_size = min_vals.shape[0]
        for batch_index in range(batch_size):
            graph = Graph()
            batch_keypoints = anchor_out[batch_index]  # (K, 4)
            keypoints = dict[int, Keypoint]()

            for kp_index, kp_data in enumerate(batch_keypoints):
                score = float(kp_data[2])
                if score >= conf_thresh:
                    label = self.label_map[int(kp_data[3])]
                    x, y = float(kp_data[0]), float(kp_data[1])
                    keypoint = Keypoint(kind=label, x=x, y=y, score=score)
                    keypoints[kp_index] = keypoint
                    graph.add(keypoint)

            batch_connections = min_inds[batch_index]
            for kp_index, conn_kp_index in enumerate(batch_connections):
                if min_vals[batch_index, kp_index]:
                    kp1 = keypoints[kp_index]
                    kp2 = keypoints[int(conn_kp_index)]
                    graph.connect(kp1, kp2)

            graphs.append(graph.resize((out_w, out_h), (in_w, in_h)))

        return graphs
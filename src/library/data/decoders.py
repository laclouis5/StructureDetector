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
        heatmaps = nms(heatmaps)  # (B, M, H/R, W/R)
        scores, inds, labels, pos = topk(heatmaps, k=self.max_objects)
        offsets = transpose_and_gather(offsets, inds)  # (B, K, 2)
        pos_kps = pos + offsets  # (B, K, 2)

        raw_keypoints = torch.stack((*pos_kps, scores, labels.float()), dim=2)  # (B, K, 4)

        # Embeddings
        embeddings = transpose_and_gather(embeddings, inds)  # (B, K, 2)
        pos_proj_kps = pos + embeddings  # (B, K, 2)

        # Association
        mask = (scores > conf_thresh).float()  # (B, K)
        scores = -(1 - mask) + mask * scores  # (B, K) 
        pos_kps = (1e6*(1 - mask) + mask * pos_kps)  # (B, K, 2)
        pos_proj_kps = (-1e6*(1 - mask) + mask * pos_proj_kps)  # (B, K, 2)

        pos_kps_mat = pos_kps.unsqueeze(2).expand(-1, -1, self.max_objects, -1)  # (B, K, K, 2)
        pos_proj_kps_mat = pos_proj_kps.unsqueeze(1).expand(-1, self.max_objects, -1, -1)  # (B, K, K, 2)

        sq_distance = torch.hypot(*torch.unbind(pos_kps_mat - pos_proj_kps_mat, dim=-1))  # (B, K, K)
        
        distances, connections = sq_distance.min(dim=1)  # (B, K)
        is_close_enough = distances < (dist_thresh * min(out_w, out_h))  # (B, K)

        # Assemble
        graphs = []
        batch_size = distances.shape[0]
        for batch_index in range(batch_size):
            graph = Graph()
            batch_keypoints = raw_keypoints[batch_index]  # (K, 4)
            keypoints = dict[int, Keypoint]()

            for kp_index, kp_data in enumerate(batch_keypoints):
                score = float(kp_data[2])
                if score >= conf_thresh:
                    label = self.label_map[int(kp_data[3])]
                    x, y = float(kp_data[0]), float(kp_data[1])
                    keypoint = Keypoint(kind=label, x=x, y=y, score=score)
                    keypoints[kp_index] = keypoint
                    graph.add(keypoint)

            batch_connections = connections[batch_index]
            for kp_index, conn_kp_index in enumerate(batch_connections):
                if is_close_enough[batch_index, kp_index]:
                    kp1 = keypoints[kp_index]
                    kp2 = keypoints[int(conn_kp_index)]
                    graph.connect(kp1, kp2)

            graphs.append(graph.resize((out_w, out_h), (in_w, in_h)))

        return graphs
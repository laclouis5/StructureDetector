from ..utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.hm_loss = FocalLoss() if args.hm_loss_fn == "focal" else nn.MSELoss()
        self.reg_loss = L1Loss()
        self.stats = LossStats()

    def forward(self, input, target):
        heatmaps = clamped_sigmoid(input["heatmaps"])

        hm_loss = self.args.hm_weight * ( \
            self.hm_loss(heatmaps, target["heatmaps"]))

        offset_loss = self.args.offset_weight * ( \
            self.reg_loss(
                input["offsets"], target["offsets"],
                target["inds"], target["off_mask"]))

        embeddings_loss = self.args.embedding_weight * self.reg_loss(
            input["embeddings"], target["embeddings"],
            target["inds"], target["emb_mask"])

        self.stats.update(hm_loss, offset_loss, embeddings_loss)

        return hm_loss + offset_loss + embeddings_loss


class L1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    # input: (B, 2, H, W), target: (B, K, 2), inds & mask: (B, K)
    def forward(self, input, target, inds, mask):
        if (numel := mask.sum()) == 0: return 0
        preds = transpose_and_gather(input, inds)  # (B, K, 2)
        loss = torch.abs((preds - target) * mask[..., None]).sum()  # (1)
        return loss / numel


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    # input: (B, 2, H, W), target: (B, K, 2), inds & mask: (B, K)
    def forward(self, input, target, inds, mask):
        preds = transpose_and_gather(input, inds)  # (B, K, 2)
        mask = mask.unsqueeze(2).expand_as(preds).float()  # (B, K, 2)
        loss = F.smooth_l1_loss(preds * mask, target * mask, reduction="sum")  # (1)
        return loss / (mask.sum() + 1e-7)  # (1)


class L2Loss(nn.Module):

    def __init__(self):
        super().__init__()

    # input: (B, 2, H, W), target: (B, K, 2), inds & mask: (B, K)
    def forward(self, input, target, inds, mask):
        preds = transpose_and_gather(input, inds)  # (B, K, 2)
        mask = mask.unsqueeze(2).expand_as(preds).float()  # (B, K, 2)
        loss = F.mse_loss(preds * mask, target * mask, reduction="sum")  # (1)
        return loss / (mask.sum() + 1e-7)  # (1)


class FocalLoss(nn.Module):

    def __init__(self):
        super().__init__()

    # input: (B, N, H, W), target: (B, N, H, W)
    def forward(self, input, target):
        pos_inds = (target == 1).float()  # (B, N, H, W)
        neg_inds = (target < 1).float()  # (B, N, H, W)

        neg_weights = torch.pow(1.0 - target, 4)  # (B, N, H, W)
        one_minus_input = 1.0 - input  # (B, N, H, W)
        
        neg_loss = torch.log(one_minus_input)  \
            * torch.pow(input, 2) \
            * neg_weights * neg_inds  # (B, N, H, W)

        num_pos = pos_inds.sum()  # (1)
        neg_loss = neg_loss.sum()  # (1)

        if num_pos == 0:
            return -neg_loss  # (1)

        pos_loss = torch.log(input) * torch.pow(one_minus_input, 2) * pos_inds  # (B, N, H, W)
        pos_loss = pos_loss.sum()  # (1)
        return -(pos_loss + neg_loss) / num_pos  # (1)


class LossStats:

    def __init__(self, hm_loss=0.0, offset_loss=0.0, embedding_loss=0.0):
        self.hm_loss = hm_loss
        self.offset_loss = offset_loss
        self.embedding_loss = embedding_loss

    def reset(self):
        self.hm_loss = 0.0
        self.offset_loss = 0.0
        self.embedding_loss = 0.0
    
    def update(self, hm_loss, offset_loss, embedding_loss):
        self.hm_loss = hm_loss
        self.offset_loss = offset_loss
        self.embedding_loss = embedding_loss

    @property
    def total_loss(self):
        return self.hm_loss + self.offset_loss + self.embedding_loss

    def __add__(self, other):
        return LossStats(
            self.hm_loss + other.hm_loss,
            self.offset_loss + other.offset_loss,
            self.embedding_loss + other.embedding_loss)

    def __iadd__(self, other):
        self.hm_loss += other.hm_loss
        self.offset_loss += other.offset_loss
        self.embedding_loss += other.embedding_loss
        return self

    def __truediv__(self, value):
        return LossStats(
            self.hm_loss / value,
            self.offset_loss / value,
            self.embedding_loss / value)
    
    def __itruediv__(self, value):
        self.hm_loss /= value
        self.offset_loss /= value
        self.embedding_loss /= value
        return self

    def __repr__(self):
        f"total_loss: {self.total_loss}, hm_loss: {self.hm_loss}, offset_loss: {self.offset_loss}, embedding_loss: {self.embedding_loss}"
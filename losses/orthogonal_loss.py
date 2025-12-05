import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.gather import GatherLayer

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def compute_barlow_loss(self, features):
        # zero-mean and normalize
        z = (features - features.mean(0)) / (features.std(0) + 1e-6)
        N = z.size(0)
        c = (z.T @ z) / N  # (D, D)

        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
        return on_diag + 0.005 * off_diag

    def forward(self, features, k, labels=None, clothes_ids=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)
        k = F.normalize(k, p=2, dim=1)

        # gather all samples from different GPUs as gallery to compute pairwise loss.
        features = torch.cat(GatherLayer.apply(features), dim=0)
        k = torch.cat(GatherLayer.apply(k), dim=0)
        labels = torch.cat(GatherLayer.apply(labels), dim=0)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        # eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)
        cloth_mask = torch.eq(clothes_ids, clothes_ids.t()).bool().to(device)
        same_clothes_mask = cloth_mask
        same_person_diff_clothes_mask = mask & (~cloth_mask)

        # mask_pos = mask.masked_fill(eye, 0).float()
        mask_pos = mask
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, k.t())

        # pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        # neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        orth_loss = ((dot_prod ** 2) * same_clothes_mask.float()).sum() / (same_clothes_mask.sum() + 1e-6)
        sim_loss = (((1 - dot_prod) ** 2) * same_person_diff_clothes_mask.float()).sum() / (same_person_diff_clothes_mask.sum() + 1e-6)

        # loss_img = self.compute_barlow_loss(features)
        # loss_txt = self.compute_barlow_loss(k)

        loss = orth_loss + sim_loss

        # loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        # loss = pos_pairs_mean + self.gamma * neg_pairs_mean

        return loss
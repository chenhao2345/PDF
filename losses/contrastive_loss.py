import torch
import torch.nn.functional as F
from torch import nn
from torch import distributed as dist
from losses.gather import GatherLayer, all_gather_with_grad


class ContrastiveLoss(nn.Module):
    """ Supervised Contrastive Learning Loss among sample pairs.

    Args:
        scale (float): scaling factor.
    """
    def __init__(self, scale=16, **kwargs):
        super().__init__()
        self.s = scale

    def forward(self, inputs, k, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # print(inputs.shape, k.shape, targets)
        # l2-normalize
        inputs = F.normalize(inputs, p=2, dim=1)
        k = F.normalize(k, p=2, dim=1)

        # gather all samples from different GPUs as gallery to compute pairwise loss.
        # gallery_inputs = torch.cat(GatherLayer.apply(k), dim=0)
        # gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        gallery_inputs = all_gather_with_grad(k)
        gallery_targets = all_gather_with_grad(targets)
        m, n = targets.size(0), gallery_targets.size(0)

        # compute cosine similarity
        similarities = torch.matmul(inputs, gallery_inputs.t()) * self.s

        # get mask for pos/neg pairs
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask = torch.eq(targets, gallery_targets.T).float().cuda()
        mask_self = torch.zeros_like(mask)
        rank = dist.get_rank()
        mask_self[:, rank * m:(rank + 1) * m] += torch.eye(m).float().cuda()
        mask_pos = mask - mask_self

        mask_neg = 1 - mask

        # compute log_prob
        exp_logits = torch.exp(similarities) * (1 - mask_self)
        # log_prob = similarities - torch.log(exp_logits.sum(1, keepdim=True))
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * mask_neg).sum(1, keepdim=True) + exp_logits)
        log_prob = similarities - log_sum_exp_pos_and_all_neg

        # compute mean of log-likelihood over positive
        loss = (mask_pos * log_prob).sum(1) / mask_pos.sum(1)

        # print(mask_self[0])
        # print(log_prob[0])
        # input()

        loss = - loss.mean()

        return loss


# class ContrastiveLoss(nn.Module):
#     def __init__(self, num_instance=8, scale=16, mode='one'):
#         super().__init__()
#         self.criterion = nn.CrossEntropyLoss()
#         self.num_instance = num_instance
#         self.T = 1.0 / scale
#         self.mode=mode
#
#     def forward(self, q, k, label):
#         batchSize = q.shape[0]
#         # l2-normalize
#         q = F.normalize(q, p=2, dim=1)
#         k = F.normalize(k, p=2, dim=1)
#         if self.mode == 'one':
#             rand_idx = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
#             # pos logit
#             l_pos = torch.bmm(q.view(batchSize, 1, -1), k[rand_idx].view(batchSize, -1, 1))
#             l_pos = l_pos.view(batchSize, 1)
#             N = q.size(0)
#             mat_sim = torch.matmul(q, k.transpose(0, 1))
#             # mat_eq = label.expand(N, N).eq(label.expand(N, N).t())
#             mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
#             # positives = torch.masked_select(mat_sim, mat_eq).view(batchSize, -1)
#             negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
#             out = torch.cat((l_pos, negatives), dim=1)/self.T
#             targets = torch.zeros([batchSize]).cuda().long()
#             loss = self.criterion(out, targets)
#         if self.mode == 'random':
#             rand_idx1 = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
#             rand_idx2 = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
#             rand_idx3 = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
#             rand_idx4 = self.get_shuffle_ids(batchSize, ranges=self.num_instance)
#
#             k = (k[rand_idx1]+k[rand_idx2]+k[rand_idx3]+k[rand_idx4])/4
#
#             # pos logit
#             l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
#             l_pos = l_pos.view(batchSize, 1)
#             N = q.size(0)
#             mat_sim = torch.matmul(q, k.transpose(0, 1))
#             # mat_eq = label.expand(N, N).eq(label.expand(N, N).t())
#             mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
#             # positives = torch.masked_select(mat_sim, mat_eq).view(batchSize, -1)
#             negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
#             out = torch.cat((l_pos, negatives), dim=1)/self.T
#             targets = torch.zeros([batchSize]).cuda().long()
#             loss = self.criterion(out, targets)
#
#         elif self.mode == 'hard':
#             N = q.size(0)
#             mat_sim = torch.matmul(q, k.transpose(0, 1))
#             mat_eq = label.expand(N, N).eq(label.expand(N, N).t()).float()
#             # batch hard
#             hard_p, hard_n, hard_p_indice, hard_n_indice = self.batch_hard(mat_sim, mat_eq, True)
#             l_pos = hard_p.view(batchSize, 1)
#             l_neg = hard_n.view(batchSize, 1)
#             mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
#             # positives = torch.masked_select(mat_sim, mat_eq).view(-1, 1)
#             negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
#             out = torch.cat((l_pos, negatives), dim=1) / self.T
#             # out = torch.cat((l_pos, l_neg, negatives), dim=1) / self.T
#             targets = torch.zeros([batchSize]).cuda().long()
#             triple_dist = F.log_softmax(out, dim=1)
#             triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)
#             # triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)*l + torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1)+1, 1) * (1-l)
#             loss = (- triple_dist_ref * triple_dist).mean(0).sum()
#
#         else:
#             N = q.size(0)
#             mat_sim = torch.matmul(q, k.transpose(0, 1))
#             mat_eq = label.expand(N, N).eq(label.expand(N, N).t())
#             mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
#             positives = torch.masked_select(mat_sim, mat_eq).view(batchSize, -1)
#             negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
#             positives = torch.topk(positives, k=self.num_instance, dim=1, largest=True, sorted=True)[0]
#             positive = torch.mean(positives[:, :self.num_instance], dim=1, keepdim=True)
#             out = torch.cat((positive, negatives), dim=1) / self.T
#             targets = torch.zeros([batchSize]).cuda().long()
#             loss = self.criterion(out, targets)
#             # loss = 0
#             # positive_range = [1, 2, 3]
#             # for i in positive_range:
#             #     out = torch.cat((positives[:,i:i+1], negatives), dim=1) / self.T
#             #     targets = torch.zeros([batchSize]).cuda().long()
#             #     loss += self.criterion(out, targets)
#             # loss = loss/len(positive_range)
#         return loss
#
#     def get_shuffle_ids(self, bsz, ranges):
#         """sample one random correct idx"""
#         rand_inds = torch.zeros(bsz).long().cuda()
#         for i in range(bsz//ranges):
#             rand_inds[i*ranges:(i+1)*ranges] = i*ranges+torch.randperm(ranges).long().cuda()
#         return rand_inds
#
#     def get_negative_ids(self, bsz, ranges):
#         """sample one random negative idx"""
#         rand_inds = torch.zeros(bsz).long().cuda()
#         for i in range(bsz//ranges):
#             rand_inds[i*ranges:(i+1)*ranges] = i*ranges+torch.randperm(ranges).long().cuda()
#         return rand_inds
#
#     def get_random_ids(self, bsz):
#         """sample one random idx"""
#         rand_inds = torch.randperm(bsz).long().cuda()
#         return rand_inds
#
#     def batch_hard(self, mat_sim, mat_eq, indice=False):
#         sorted_mat_sim, positive_indices = torch.sort(mat_sim + (9999999.) * (1 - mat_eq), dim=1,
#                                                            descending=False)
#         hard_p = sorted_mat_sim[:, 0]
#         hard_p_indice = positive_indices[:, 0]
#         sorted_mat_distance, negative_indices = torch.sort(mat_sim + (-9999999.) * (mat_eq), dim=1,
#                                                            descending=True)
#         hard_n = sorted_mat_distance[:, 0]
#         hard_n_indice = negative_indices[:, 0]
#         if (indice):
#             return hard_p, hard_n, hard_p_indice, hard_n_indice
#         return hard_p, hard_n


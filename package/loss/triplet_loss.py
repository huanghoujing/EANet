from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
from ..eval.torch_distance import compute_dist
from ..utils.meter import RecentAverageMeter as Meter
from .loss import Loss


class _TripletLoss(object):
    """Reference:
        https://github.com/Cysu/open-reid
        In Defense of the Triplet Loss for Person Re-Identification
    """
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an):
        """
        Args:
          dist_ap: pytorch tensor, distance between anchor and positive sample, shape [N]
          dist_an: pytorch tensor, distance between anchor and negative sample, shape [N]
        Returns:
          loss: pytorch scalar
        """
        y = torch.ones_like(dist_ap)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


def construct_triplets(dist_mat, labels, hard_type='tri_hard'):
    """Construct triplets inside a batch.
    Args:
        dist_mat: pytorch tensor, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
    Returns:
        dist_ap: pytorch tensor, distance(anchor, positive); shape [M]
        dist_an: pytorch tensor, distance(anchor, negative); shape [M]
    NOTE: Only consider PK batch, so we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_self = labels.new().resize_(N, N).copy_(torch.eye(N)).byte()
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    K = is_pos.sum(1)[0]
    P = int(N / K)
    assert P * K == N, "P * K = {}, N = {}".format(P * K, N)
    is_pos = ~ is_self & is_pos
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    if hard_type == 'semi':
        dist_ap = dist_mat[is_pos].contiguous().view(-1)
        dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        dist_an = dist_an.expand(N, K - 1).contiguous().view(-1)
    elif hard_type == 'all':
        dist_ap = dist_mat[is_pos].contiguous().view(N, K - 1).unsqueeze(-1).expand(N, K - 1, P * K - K).contiguous().view(-1)
        dist_an = dist_mat[is_neg].contiguous().view(N, P * K - K).unsqueeze(1).expand(N, K - 1, P * K - K).contiguous().view(-1)
    elif hard_type == 'tri_hard':
        # `dist_ap` means distance(anchor, positive); both `dist_ap` and `relative_p_inds` with shape [N]
        dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=False)
        # `dist_an` means distance(anchor, negative); both `dist_an` and `relative_n_inds` with shape [N]
        dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=False)
    else:
        raise NotImplementedError

    # print("dist_ap.size() {}, dist_an.size() {}, N {}, P {}, K {}".format(dist_ap.size(), dist_an.size(), N, P, K))
    assert dist_ap.size() == dist_an.size(), "dist_ap.size() {}, dist_an.size() {}".format(dist_ap.size(), dist_an.size())
    return dist_ap, dist_an


class TripletLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(TripletLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.tri_loss_obj = _TripletLoss(margin=cfg.margin)

    def calculate(self, feat, labels, hard_type=None):
        """
        Args:
            feat: pytorch tensor, shape [N, C]
            labels: pytorch LongTensor, with shape [N]
            hard_type: can be dynamically set to different types during training, for hybrid or curriculum learning
        Returns:
            loss: pytorch scalar
            ==================
            For Debugging, etc
            ==================
            dist_ap: pytorch tensor, distance(anchor, positive); shape [N]
            dist_an: pytorch tensor, distance(anchor, negative); shape [N]
            dist_mat: pytorch tensor, pairwise euclidean distance; shape [N, N]
        """
        cfg = self.cfg
        dist_mat = compute_dist(feat, feat, dist_type=cfg.dist_type)
        if hard_type is None:
            hard_type = cfg.hard_type
        dist_ap, dist_an = construct_triplets(dist_mat, labels, hard_type=hard_type)
        loss = self.tri_loss_obj(dist_ap, dist_an)
        if cfg.norm_by_num_of_effective_triplets:
            sm = (dist_an > dist_ap + cfg.margin).float().mean().item()
            loss *= 1. / (1 - sm + 1e-8)
        return {'loss': loss, 'dist_ap': dist_ap, 'dist_an': dist_an}

    def __call__(self, batch, pred, step=0, hard_type=None):
        # NOTE: Here is only a trial implementation for PAP-9P*

        # Calculation
        cfg = self.cfg
        res1 = self.calculate(torch.cat(pred['feat_list'][:6], 1), batch['label'], hard_type=hard_type)
        res2 = self.calculate(torch.cat(pred['feat_list'][7:9], 1), batch['label'], hard_type=hard_type)
        res3 = self.calculate(pred['feat_list'][9], batch['label'], hard_type=hard_type)
        loss = res1['loss'] + res2['loss'] + res3['loss']

        # Meter
        if cfg.name not in self.meter_dict:
            self.meter_dict[cfg.name] = Meter(name=cfg.name)
        self.meter_dict[cfg.name].update(loss.item())
        dist_an = res1['dist_an']
        dist_ap = res1['dist_ap']
        # precision
        key = 'prec'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        self.meter_dict[key].update((dist_an > dist_ap).float().mean().item())
        # the proportion of triplets that satisfy margin
        key = 'sm'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key, fmt='{:.2%}')
        self.meter_dict[key].update((dist_an > dist_ap + cfg.margin).float().mean().item())
        # average (anchor, positive) distance
        key = 'd_ap'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key)
        self.meter_dict[key].update(dist_ap.mean().item())
        # average (anchor, negative) distance
        key = 'd_an'
        if key not in self.meter_dict:
            self.meter_dict[key] = Meter(name=key)
        self.meter_dict[key].update(dist_an.mean().item())

        # Tensorboard
        if self.tb_writer is not None:
            for key in [cfg.name, 'prec', 'sm', 'd_ap', 'd_an']:
                self.tb_writer.add_scalars(key, {key: self.meter_dict[key].avg}, step)

        # Scale by loss weight
        loss *= cfg.weight

        return {'loss': loss}

import torch.nn as nn


class PCBPool(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pool = nn.AdaptiveAvgPool2d(1) if cfg.max_or_avg == 'avg' else nn.AdaptiveMaxPool2d(1)

    def __call__(self, in_dict):
        feat = in_dict['feat']
        assert feat.size(2) % self.cfg.num_parts == 0
        stripe_h = int(feat.size(2) / self.cfg.num_parts)
        feat_list = []
        for i in range(self.cfg.num_parts):
            # shape [N, C]
            local_feat = self.pool(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :])
            # shape [N, C]
            local_feat = local_feat.view(local_feat.size(0), -1)
            feat_list.append(local_feat)
        out_dict = {'feat_list': feat_list}
        return out_dict

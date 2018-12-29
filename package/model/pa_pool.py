import torch.nn.functional as F


def pa_avg_pool(in_dict):
    """Mask weighted avg pooling.
    Args:
        feat: pytorch tensor, with shape [N, C, H, W]
        mask: pytorch tensor, with shape [N, pC, pH, pW]
    Returns:
        feat_list: a list (length = pC) of pytorch tensors with shape [N, C]
        visible: pytorch tensor with shape [N, pC]
    """
    feat = in_dict['feat']
    mask = in_dict['pap_mask']
    N, C, H, W = feat.size()
    N, pC, pH, pW = mask.size()
    # 1 * [N, C, pH, pW] -> [N, 1, C, pH, pW] -> [N, pC, C, pH, pW]
    feat = feat.unsqueeze(1).expand((N, pC, C, pH, pW))
    # [N, pC]
    visible = (mask.sum(-1).sum(-1) != 0).float()
    # [N, pC, 1, pH, pW] -> [N, pC, C, pH, pW]
    mask = mask.unsqueeze(2).expand((N, pC, C, pH, pW))
    # [N, pC, C]
    feat = (feat * mask).sum(-1).sum(-1) / (mask.sum(-1).sum(-1) + 1e-12)
    # pC * [N, C]
    feat_list = list(feat.transpose(0, 1))
    out_dict = {'feat_list': feat_list, 'visible': visible}
    return out_dict


def pa_max_pool(in_dict):
    """Implement `local max pooling` as `masking + global max pooling`.
    Args:
        feat: pytorch tensor, with shape [N, C, H, W]
        mask: pytorch tensor, with shape [N, pC, pH, pW]
    Returns:
        feat_list: a list (length = pC) of pytorch tensors with shape [N, C]
        visible: pytorch tensor with shape [N, pC]
    NOTE:
        The implementation of `masking + global max pooling` is only equivalent
        to `local max pooling` when feature values are non-negative, which holds
        for ResNet that has ReLU as final operation of all blocks.
    """
    feat = in_dict['feat']
    mask = in_dict['pap_mask']
    N, C, H, W = feat.size()
    N, pC, pH, pW = mask.size()
    feat_list = []
    for i in range(pC):
        # [N, C, pH, pW]
        m = mask[:, i, :, :].unsqueeze(1).expand_as(feat)
        # [N, C]
        local_feat = F.adaptive_max_pool2d(feat * m, 1).view(N, -1)
        feat_list.append(local_feat)
    # [N, pC]
    visible = (mask.sum(-1).sum(-1) != 0).float()
    out_dict = {'feat_list': feat_list, 'visible': visible}
    return out_dict


class PAPool(object):
    def __init__(self, cfg):
        self.pool = pa_avg_pool if cfg.max_or_avg == 'avg' else pa_max_pool

    def __call__(self, *args, **kwargs):
        return self.pool(*args, **kwargs)

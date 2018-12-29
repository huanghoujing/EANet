import torch


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch tensor
    Returns:
        x: pytorch tensor, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch tensor, with shape [m, d]
        y: pytorch tensor, with shape [n, d]
    Returns:
        dist: pytorch tensor, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def compute_dist(array1, array2, dist_type='cosine', cos_to_normalize=True):
    """
    Args:
        array1: pytorch tensor, with shape [m, d]
        array2: pytorch tensor, with shape [n, d]
    Returns:
        dist: pytorch tensor, with shape [m, n]
    """
    if dist_type == 'cosine':
        if cos_to_normalize:
            array1 = normalize(array1, axis=1)
            array2 = normalize(array2, axis=1)
        dist = - torch.mm(array1, array2.t())
        # Turn distance into positive value
        dist += 1
    elif dist_type == 'euclidean':
        dist = euclidean_dist(array1, array2)
    else:
        raise NotImplementedError
    return dist

import numpy as np
from PIL import Image
import cv2
from os.path import dirname as ospdn
from .file import may_make_dir


def make_im_grid(ims, n_rows, n_cols, space, pad_val):
    """Make a grid of images with space in between.
    Args:
      ims: a list of [3, im_h, im_w] images
      n_rows: num of rows
      n_cols: num of columns
      space: the num of pixels between two images
      pad_val: scalar, or numpy array with shape [3]; the color of the space
    Returns:
      ret_im: a numpy array with shape [3, H, W]
    """
    assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
    if (n_rows is None) and (n_cols is None):
        n_cols = int(np.ceil(np.sqrt(len(ims))))
        n_rows = int(np.ceil(1. * len(ims) / n_cols))
    else:
        assert len(ims) <= n_rows * n_cols
    h, w = ims[0].shape[1:]
    H = h * n_rows + space * (n_rows - 1)
    W = w * n_cols + space * (n_cols - 1)
    if isinstance(pad_val, np.ndarray):
        # reshape to [3, 1, 1]
        pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
    ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)
    for n, im in enumerate(ims):
        r = n // n_cols
        c = n % n_cols
        h1 = r * (h + space)
        h2 = r * (h + space) + h
        w1 = c * (w + space)
        w2 = c * (w + space) + w
        ret_im[:, h1:h2, w1:w2] = im
    return ret_im


def read_im(im_path, convert_rgb=True, resize_h_w=(128, 64), transpose=True):
    im = Image.open(im_path)
    if convert_rgb:
        # shape [H, W, 3]
        im = im.convert("RGB")
    im = np.asarray(im)
    if resize_h_w is not None and (im.shape[0], im.shape[1]) != resize_h_w:
        im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    if transpose:
        # shape [3, H, W]
        im = im.transpose(2, 0, 1)
    return im


def save_im(im, save_path, transpose=False, check_bound=False):
    """
    im: (1) shape [3, H, W], transpose should be True
        (2) shape [H, W, 3], transpose should be False
        (3) shape [H, W], transpose should be False
    """
    may_make_dir(ospdn(save_path))
    if transpose:
        im = im.transpose(1, 2, 0)
    if check_bound:
        im = im.clip(0, 255)
    im = im.astype(np.uint8)
    mode = 'L' if len(im.shape) == 2 else 'RGB'
    im = Image.fromarray(im, mode=mode)
    im.save(save_path)


def heatmap_to_color_im(
    hmap,
    normalize=False, min_max_val=None,
    resize=False, resize_w_h=None,
    transpose=False
):
    """
    Args:
        hmap: a numpy array with shape [h, w]
        normalize: whether to normalize the value to range [0, 1]. If `False`,
            make sure that `hmap` has been in range [0, 1]
    Return:
        hmap: shape [h, w, 3] if transpose=False, shape [3, h, w] if transpose=True, with value in range [0, 255], uint8"""
    if resize:
        hmap = cv2.resize(hmap, tuple(resize_w_h), interpolation=cv2.INTER_LINEAR)
    # normalize to interval [0, 1]
    if normalize:
        if min_max_val is None:
            min_v, max_v = np.min(hmap), np.max(hmap)
        else:
            min_v, max_v = min_max_val
        hmap = (hmap - min_v) / (float(max_v - min_v) + 1e-8)
    # The `cv2.applyColorMap(gray_im, cv2.COLORMAP_JET)` maps 0 to RED and 1
    # to BLUE, not normal. So rectify it.
    hmap = 1 - hmap
    hmap = (hmap * 255).clip(0, 255).astype(np.uint8)
    # print(hmap.shape, hmap.dtype, np.min(hmap), np.max(hmap))
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    if transpose:
        hmap = hmap.transpose(2, 0, 1)
    return hmap


def restore_im(im, std, mean, transpose=False, resize_w_h=None):
    """Invert the normalization process.
    Args:
        im: normalized im with shape [3, h, w]
    Returns:
        im: shape [h, w, 3] if transpose=True, shape [3, h, w] if transpose=False, with value in range [0, 255], uint8
    """
    im = im * np.array(std)[:, np.newaxis, np.newaxis]
    im = im + np.array(mean)[:, np.newaxis, np.newaxis]
    im = (im * 255).clip(0, 255).astype(np.uint8)
    if resize_w_h is not None:
        im = cv2.resize(im.transpose(1, 2, 0), tuple(resize_w_h), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    if transpose:
        im = np.transpose(im, [1, 2, 0])
    return im

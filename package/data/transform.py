import torch
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image
import cv2

"""We expect a list `cfg.transform_list`. The types specified in this list 
will be applied sequentially. Each type name corresponds to a function name in 
this file, so you have to implement the function w.r.t. your custom type. 
The function head should be `FUNC_NAME(in_dict, cfg)`, and it should modify `in_dict`
in place.
The transform list allows us to apply optional transforms in any order, while custom
functions allow us to perform sync transformation for images and all labels.
"""


def hflip(in_dict, cfg):
    # Tricky!! random.random() can not reproduce the score of np.random.random(),
    # dropping ~1% for both Market1501 and Duke GlobalPool.
    # if random.random() < 0.5:
    if np.random.random() < 0.5:
        in_dict['im'] = F.hflip(in_dict['im'])
        if 'pap_mask' in in_dict:
            in_dict['pap_mask'] = in_dict['pap_mask'][:, :, ::-1]
        if 'ps_label' in in_dict:
            in_dict['ps_label'] = F.hflip(in_dict['ps_label'])


def resize_3d_np_array(maps, resize_h_w, interpolation):
    """maps: np array with shape [C, H, W], dtype is not restricted"""
    return np.stack([cv2.resize(m, tuple(resize_h_w[::-1]), interpolation=interpolation) for m in maps])


# def may_resize(im, resize_h_w, interpolation):
#     """im: PIL image"""
#     resize_w_h = tuple(resize_h_w[::-1])
#     if im.size != resize_w_h:
#         # Refer to
#         #     from PIL import Image
#         #     Image.Image.resize()
#         im = im.resize(resize_w_h, resample=interpolation)
#     return im


# Resize image using PIL.Image.Image.resize()
# def resize(in_dict, cfg):
#     in_dict['im'] = may_resize(in_dict['im'], cfg.im.h_w, Image.BILINEAR)
#     if 'pap_mask' in in_dict:
#         in_dict['pap_mask'] = resize_3d_np_array(in_dict['pap_mask'], cfg.pap_mask.h_w, cv2.INTER_NEAREST)
#     if 'ps_label' in in_dict:
#         in_dict['ps_label'] = may_resize(in_dict['ps_label'], cfg.ps_label.h_w, Image.NEAREST)


# Resize image using cv2.resize()
def resize(in_dict, cfg):
    in_dict['im'] = Image.fromarray(cv2.resize(np.array(in_dict['im']), tuple(cfg.im.h_w[::-1]), interpolation=cv2.INTER_LINEAR))
    if 'pap_mask' in in_dict:
        in_dict['pap_mask'] = resize_3d_np_array(in_dict['pap_mask'], cfg.pap_mask.h_w, cv2.INTER_NEAREST)
    if 'ps_label' in in_dict:
        in_dict['ps_label'] = Image.fromarray(cv2.resize(np.array(in_dict['ps_label']), tuple(cfg.ps_label.h_w[::-1]), cv2.INTER_NEAREST), mode='L')


def to_tensor(in_dict, cfg):
    in_dict['im'] = F.to_tensor(in_dict['im'])
    in_dict['im'] = F.normalize(in_dict['im'], cfg.im.mean, cfg.im.std)
    if 'pap_mask' in in_dict:
        in_dict['pap_mask'] = torch.from_numpy(in_dict['pap_mask']).float()
    if 'ps_label' in in_dict:
        in_dict['ps_label'] = torch.from_numpy(np.array(in_dict['ps_label'])).long()


def transform(in_dict, cfg):
    for t in cfg.transform_list:
        eval('{}(in_dict, cfg)'.format(t))
    to_tensor(in_dict, cfg)
    return in_dict

from __future__ import print_function
from tqdm import tqdm
import torch
from .torch_distance import normalize
from ..utils.misc import concat_dict_list
from ..utils.torch_utils import recursive_to_device


def extract_batch_feat(model, in_dict, cfg):
    model.eval()
    with torch.no_grad():
        in_dict = recursive_to_device(in_dict, cfg.device)
        out_dict = model(in_dict, forward_type=cfg.forward_type)
        out_dict['feat_list'] = [normalize(f) for f in out_dict['feat_list']]
        feat = torch.cat(out_dict['feat_list'], 1)
        feat = feat.cpu().numpy()
        ret_dict = {
            'im_path': in_dict['im_path'],
            'feat': feat,
        }
        if 'label' in in_dict:
            ret_dict['label'] = in_dict['label'].cpu().numpy()
        if 'cam' in in_dict:
            ret_dict['cam'] = in_dict['cam'].cpu().numpy()
        if 'visible' in out_dict:
            ret_dict['visible'] = out_dict['visible'].cpu().numpy()
    return ret_dict


def extract_dataloader_feat(model, loader, cfg):
    dict_list = []
    for batch in tqdm(loader, desc='Extract Feature', miniters=20, ncols=120, unit=' batches'):
        feat_dict = extract_batch_feat(model, batch, cfg)
        dict_list.append(feat_dict)
    ret_dict = concat_dict_list(dict_list)
    return ret_dict

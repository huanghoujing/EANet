from __future__ import print_function
import numpy as np
from .metric import cmc, mean_ap
from .np_distance import compute_dist, compute_dist_with_visibility
from ..utils.log import score_str


def get_scores_str(mAP, cmc_scores, score_prefix):
    return score_prefix + '[mAP: {}], [cmc1: {}], [cmc5: {}], [cmc10: {}]'.format(
        score_str(mAP), score_str(cmc_scores[0]), score_str(cmc_scores[4]), score_str(cmc_scores[9]))


def eval_feat(dic, cfg):
    mAP = 0
    cmc_scores = 0
    num = 0
    # Split query set into chunks, so that computing distance matrix does not run out of memory.
    chunk_size = cfg.chunk_size
    num_chunks = int(np.ceil(1. * len(dic['q_feat']) / chunk_size))
    for i in range(num_chunks):
        st_ind = i * chunk_size
        end_ind = min(i * chunk_size + chunk_size, len(dic['q_feat']))
        if 'q_visible' in dic:
            dist_mat = compute_dist_with_visibility(dic['q_feat'][st_ind:end_ind], dic['g_feat'], dic['q_visible'][st_ind:end_ind], dic['g_visible'])
        else:
            dist_mat = compute_dist(dic['q_feat'][st_ind:end_ind], dic['g_feat'])
        # Modifying distance matrix may be used in the future.
        if 'add_dist_mat' in dic:
            add_dist_mat_ = dic['add_dist_mat'][st_ind:end_ind]
            assert add_dist_mat_.shape == dist_mat.shape
            dist_mat += add_dist_mat_
        input_kwargs = dict(
            distmat=dist_mat,
            query_ids=dic['q_label'][st_ind:end_ind],
            gallery_ids=dic['g_label'],
            query_cams=dic['q_cam'][st_ind:end_ind],
            gallery_cams=dic['g_cam'],
        )
        # Compute mean AP
        mAP_ = mean_ap(**input_kwargs)
        # Compute CMC scores
        cmc_scores_ = cmc(
            separate_camera_set=cfg.separate_camera_set,
            single_gallery_shot=cfg.single_gallery_shot,
            first_match_break=cfg.first_match_break,
            topk=10,
            **input_kwargs
        )
        n = end_ind - st_ind
        mAP += mAP_ * n
        cmc_scores += cmc_scores_ * n
        num += n
    mAP /= num
    cmc_scores /= num
    scores_str = get_scores_str(mAP, cmc_scores, cfg.score_prefix)
    print(scores_str)
    return {
        'mAP': mAP,
        'cmc_scores': cmc_scores,
        'scores_str': scores_str,
    }

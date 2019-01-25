"""Visualize retrieving results. This script demonstrates how to use a trained model for inference."""
from __future__ import print_function


import sys
sys.path.insert(0, '.')
import os
import argparse
import numpy as np
from package.utils.file import walkdir
from package.utils.arg_parser import str2bool
from package.optim.eanet_trainer import EANetTrainer
from package.eval.np_distance import compute_dist, compute_dist_with_visibility
from package.utils.rank_list import get_rank_list, save_rank_list_to_im

# NOTE: If you use id and camera info, you have to define parse_im_path for your images
# E.g.
from package.data.datasets.market1501 import Market1501
parse_im_path = Market1501.parse_im_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_im_dir', type=str, default='None', help='Directory of query images.')
    parser.add_argument('--g_im_dir', type=str, default='None', help='Directory of gallery images.')
    parser.add_argument('--num_queries', type=int, default=16, help='How many query images to visualize.')
    parser.add_argument('--rank_list_size', type=int, default=10, help='How many top gallery images to visualize for each query.')
    parser.add_argument('--save_dir', type=str, default='None', help='Where to save visualization result.')
    parser.add_argument('--id_aware', type=str2bool, default=False,
                        help='Whether id and camera info are known for your images. If known, each gallery image in '
                             'rank list will have either green or red boundary, denoting same or different id as query.')
    parser.add_argument('--pap_mask_provided', type=str2bool, default=False,
                        help='Do you provide pap_mask or not. If not, all PAP based models are not available '
                             'for inference, and you can only use GlobalPool and PCB.')
    args, _ = parser.parse_known_args()
    return args


def main(args):
    q_im_paths = sorted(list(walkdir(args.q_im_dir, exts=['.jpg', '.png'])))
    q_im_paths = np.random.RandomState(1).choice(q_im_paths, args.num_queries)
    g_im_paths = sorted(list(walkdir(args.g_im_dir, exts=['.jpg', '.png'])))
    trainer = EANetTrainer()
    ###################
    # NOTE: If you want to use Part Aligned Pooling, you have to provide pap_mask. E.g.
    market1501_loader = trainer.create_dataloader('test', 'market1501', 'query')
    def get_pap_mask(im_path):
        return market1501_loader.dataset.get_pap_mask('/'.join(im_path.split('/')[2:]))
    ###################
    q_feat = trainer.infer_im_list(q_im_paths, get_pap_mask=get_pap_mask if args.pap_mask_provided else None)
    g_feat = trainer.infer_im_list(g_im_paths, get_pap_mask=get_pap_mask if args.pap_mask_provided else None)
    if 'visible' in q_feat:
        dist_mat = compute_dist_with_visibility(q_feat['feat'], g_feat['feat'], q_feat['visible'], g_feat['visible'])
    else:
        dist_mat = compute_dist(q_feat['feat'], g_feat['feat'])
    g_ids, g_cams = zip(*[parse_im_path(g_im_path) for g_im_path in g_im_paths]) if args.id_aware else (None, None)
    for i, (dist_vec, q_im_path) in enumerate(zip(dist_mat, q_im_paths)):
        q_id, q_cam = parse_im_path(q_im_path) if args.id_aware else (None, None)
        rank_list, same_id = get_rank_list(dist_vec, q_id, q_cam, g_ids, g_cams, args.rank_list_size, id_aware=args.id_aware)
        save_rank_list_to_im(rank_list, q_im_path, g_im_paths, os.path.join(args.save_dir, q_im_path.replace('/', '_')), same_id=same_id)


if __name__ == '__main__':
    main(parse_args())

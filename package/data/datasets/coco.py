from __future__ import print_function
import os.path as osp
from os.path import basename as ospbn
import numpy as np
import itertools
from ..dataset import Dataset
from ...utils.file import get_files_by_pattern
from ...utils.file import save_pickle, load_pickle


kpt_names = [
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]
kpt_name_to_ind = dict(zip(kpt_names, range(len(kpt_names))))


class COCO(Dataset):
    has_pap_mask = False
    has_ps_label = True
    im_root = 'images'
    split_spec = {
        'train': {'patterns': ['{}/train/*.jpg'.format(im_root)]},
        'val': {'patterns': ['{}/val/*.jpg'.format(im_root)]},
        'train_market1501_style': {'patterns': ['{}/train/*.jpg'.format(im_root), '{}/train_market1501_style/*.jpg'.format(im_root)]},  # Original COCO images are also used in paper's PAP-StC-PS
        'train_cuhk03_style': {'patterns': ['{}/train/*.jpg'.format(im_root), '{}/train_cuhk03_style/*.jpg'.format(im_root)]},
        'train_duke_style': {'patterns': ['{}/train/*.jpg'.format(im_root), '{}/train_duke_style/*.jpg'.format(im_root)]},
    }

    def get_pap_mask(self, im_path):
        err_msg = 'If you want to generate pap mask from keypoint for COCO, you have to ' \
                  'deal with the file structure difference between im_path_to_kpt.pkl and ' \
                  'im_name_to_kpt.pkl. So implement it here.'
        raise NotImplementedError(err_msg)

    def ratio_ok(self, kpts, im_h_w):
        """Select those samples (1) with nearly full body and (2) in upright pose, so that
        we can lazily resize them as for ReID images without over distorting body ratio.
        """
        def _any_visible(kpts, kpt_names):
            inds = [kpt_name_to_ind[n] for n in kpt_names]
            kpts = kpts[inds]
            return np.max(kpts[:, 2]) > 0
        main_groups = [['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
                       ['left_shoulder', 'right_shoulder'],
                       ['left_hip', 'right_hip'],
                       ['left_knee', 'right_knee'],
                       ['left_ankle', 'right_ankle'],
                       ]
        # n_visible_kpts = np.sum(kpts[:, 2] > 0)
        n_visible_groups = np.sum([_any_visible(kpts, g) for g in main_groups])
        h, w = im_h_w
        return (n_visible_groups >= 4) and (1. * h / w > 1.8)

    def filter_ratio(self, im_paths):
        im_name_to_kpt = load_pickle(osp.join(self.root, 'im_name_to_kpt.pkl'))
        im_name_to_h_w = load_pickle(osp.join(self.root, 'im_name_to_h_w.pkl'))
        im_paths = [imp for imp in im_paths if self.ratio_ok(im_name_to_kpt[ospbn(imp)], im_name_to_h_w[ospbn(imp)])]
        return im_paths

    def save_split(self, spec, save_path):
        cfg = self.cfg
        im_paths = sorted(itertools.chain.from_iterable(
            [get_files_by_pattern(self.root, pattern=p, strip_root=True)
             for p in spec['patterns']]
        ))
        # TODO: Don't filter in the future, so that we can perform standard segmentation training.
        print('Before filtering: Split {} of {} has {} images'.format(cfg.split, self.__class__.__name__, len(im_paths)))
        im_paths = self.filter_ratio(im_paths)
        print('After filtering: Split {} of {} has {} images'.format(cfg.split, self.__class__.__name__, len(im_paths)))
        samples = [{'im_path': im_path} for im_path in im_paths]
        save_pickle(samples, save_path)

    def _get_ps_label_path(self, im_path):
        cfg = self.cfg
        path = im_path
        # Style transferred COCO images use the same ps labels as original COCO images.
        splits = ['train_market1501_style', 'train_cuhk03_style', 'train_duke_style']
        for split in splits:
            path = path.replace(split, 'train')
        path = path.replace(self.im_root, 'masks_7_parts')
        path = path.replace('.jpg', '.png')
        path = osp.join(self.root, path)
        return path

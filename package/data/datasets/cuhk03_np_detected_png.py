import os.path as osp
from .market1501 import Market1501


class CUHK03NpDetectedPng(Market1501):
    # Will be available in the future
    has_pap_mask = False
    has_ps_label = False
    im_root = 'cuhk03-np'
    split_spec = {
        'train': {'pattern': '{}/detected/bounding_box_train/*.png'.format(im_root), 'map_label': True},
        'query': {'pattern': '{}/detected/query/*.png'.format(im_root), 'map_label': False},
        'gallery': {'pattern': '{}/detected/bounding_box_test/*.png'.format(im_root), 'map_label': False},
    }

    def _get_kpt_key(self, im_path):
        return im_path

    def _get_ps_label_path(self, im_path):
        return osp.join(self.root, im_path.replace(self.im_root, self.im_root + '_ps_label'))

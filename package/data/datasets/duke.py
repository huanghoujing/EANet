import os.path as osp
from .market1501 import Market1501


class DukeMTMCreID(Market1501):
    has_pap_mask = True
    has_ps_label = True
    im_root = 'DukeMTMC-reID'
    split_spec = {
        'train': {'pattern': '{}/bounding_box_train/*.jpg'.format(im_root), 'map_label': True},
        'query': {'pattern': '{}/query/*.jpg'.format(im_root), 'map_label': False},
        'gallery': {'pattern': '{}/bounding_box_test/*.jpg'.format(im_root), 'map_label': False},
        'train_market1501_style': {'pattern': 'bounding_box_train_market1501_style/*.jpg', 'map_label': True},
    }

    def _get_kpt_key(self, im_path):
        cfg = self.cfg
        key = im_path
        if cfg.split == 'train_market1501_style':
            key = '{}/bounding_box_train/{}'.format(self.im_root, osp.basename(im_path))
        return key

    def _get_ps_label_path(self, im_path):
        cfg = self.cfg
        if cfg.split == 'train_market1501_style':
            path = '{}_ps_label/bounding_box_train/{}'.format(self.im_root, osp.basename(im_path))
        else:
            path = im_path.replace(self.im_root, self.im_root + '_ps_label')
        path = path.replace('.jpg', '.png')
        path = osp.join(self.root, path)
        return path

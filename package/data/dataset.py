from __future__ import print_function
import os.path as osp
from PIL import Image
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset as TorchDataset
from .transform import transform
from ..utils.file import load_pickle


class Dataset(TorchDataset):
    """Args:
        samples: None or a list of dicts; samples[i] has key 'im_path' and optional 'label', 'cam'.
    """
    has_pap_mask = None
    has_ps_label = None
    im_root = None
    split_spec = None

    def __init__(self, cfg, samples=None):
        self.cfg = cfg
        self.root = osp.join(cfg.root, cfg.name)
        if samples is None:
            self.samples = self.load_split()
        else:
            self.samples = samples
            cfg.split = 'None'
        print(self.summary)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cfg = self.cfg
        # Deepcopy to inherit all meta items
        sample = deepcopy(self.samples[index])
        im_path = sample['im_path']
        sample['im'] = self.get_im(im_path)
        # TODO: Precisely set use_pap_mask and use_ps_label at outside scope at run time
        if self.has_pap_mask and cfg.use_pap_mask:
            sample['pap_mask'] = self.get_pap_mask(im_path)
        if self.has_ps_label and cfg.use_ps_label:
            sample['ps_label'] = self.get_ps_label(im_path)
        transform(sample, cfg)
        return sample

    def save_split(self, spec, save_path):
        raise NotImplementedError

    def load_split(self):
        cfg = self.cfg
        save_path = osp.join(self.root, cfg.split + '.pkl')
        self.save_split(self.split_spec[cfg.split], save_path)
        samples = load_pickle(save_path)
        return samples

    def get_im(self, im_path):
        return Image.open(osp.join(self.root, im_path)).convert("RGB")

    def _get_kpt_key(self, im_path):
        raise NotImplementedError

    def get_pap_mask(self, im_path):
        # Place here to avoid circular import, since
        #     dataset.py imports kpt_to_pap_mask.py,
        #     kpt_to_pap_mask.py imports coco.py,
        #     coco.py imports dataset.py
        # About circular import, this is an excellent tutorial: https://stackabuse.com/python-circular-imports
        from .kpt_to_pap_mask import gen_pap_masks
        cfg = self.cfg
        if not hasattr(self, 'im_path_to_kpt'):
            self.im_path_to_kpt = load_pickle(osp.join(self.root, 'im_path_to_kpt.pkl'))
        key = self._get_kpt_key(im_path)
        kpt = self.im_path_to_kpt[key]['kpt']
        kpt[:, 2] = (kpt[:, 2] > 0.1).astype(np.float)
        pap_mask, _ = gen_pap_masks(self.im_path_to_kpt[key]['im_h_w'], cfg.pap_mask.h_w, kpt, mask_type=cfg.pap_mask.type)
        return pap_mask

    def _get_ps_label_path(self, im_path):
        raise NotImplementedError

    def get_ps_label(self, im_path):
        cfg = self.cfg
        ps_label = Image.open(self._get_ps_label_path(im_path))
        return ps_label

    # Use property (instead of setting it in self.__init__) in case self.samples is changed after initialization.
    @property
    def num_samples(self):
        return len(self.samples)

    @property
    def num_ids(self):
        return len(set([s['label'] for s in self.samples])) if 'label' in self.samples[0] else -1

    @property
    def num_cams(self):
        return len(set([s['cam'] for s in self.samples])) if 'cam' in self.samples[0] else -1

    @property
    def summary(self):
        summary = ['=' * 25]
        summary += [self.__class__.__name__]
        summary += ['=' * 25]
        summary += ['   split: {}'.format(self.cfg.split)]
        summary += ['# images: {}'.format(self.num_samples)]
        summary += ['   # ids: {}'.format(self.num_ids)]
        summary += ['  # cams: {}'.format(self.num_cams)]
        summary = '\n'.join(summary) + '\n'
        return summary

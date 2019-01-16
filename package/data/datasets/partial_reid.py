import os.path as osp
from ..dataset import Dataset
from ...utils.file import get_files_by_pattern
from ...utils.file import save_pickle


class PartialREID(Dataset):
    # Will be available in the future
    has_pap_mask = False
    has_ps_label = False
    im_root = 'Partial-REID_Dataset'
    split_spec = {
            'occ_query': {'pattern': '{}/occluded_body_images/*.jpg'.format(im_root), 'map_label': False},
            'partial_query': {'pattern': '{}/partial_body_images/*.jpg'.format(im_root), 'map_label': False},
            # Paper "Deep Spatial Feature Reconstruction for Partial Person Re-Identification: Alignment-Free Approach" uses cropped query images
            'query': {'pattern': '{}/partial_body_images/*.jpg'.format(im_root), 'map_label': False},
            'gallery': {'pattern': '{}/whole_body_images/*.jpg'.format(im_root), 'map_label': False},
    }

    def _get_kpt_key(self, im_path):
        return im_path

    def _get_ps_label_path(self, im_path):
        return osp.join(self.root, im_path.replace(self.im_root, self.im_root + '_ps_label').replace('.jpg', '.png'))

    @staticmethod
    def parse_im_path(im_path):
        im_name = osp.basename(im_path)
        id = int(im_name[:3])
        return id

    def save_split(self, spec, save_path):
        cfg = self.cfg
        im_paths = sorted(get_files_by_pattern(self.root, pattern=spec['pattern'], strip_root=True))
        assert len(im_paths) > 0, "There are {} images for split [{}] of dataset [{}]. Please place your dataset in right position." \
            .format(len(im_paths), cfg.split, self.__class__.__name__)
        # The dataset does not annotate camera. Here we manually set query camera to 0, gallery to 1, to satisfy the testing code.
        ids = [self.parse_im_path(p) for p in im_paths]
        cams = [0 if cfg.split in ['query', 'occ_query', 'partial_query'] else 1 for _ in im_paths]
        if spec['map_label']:
            unique_ids = sorted(list(set(ids)))
            ids2labels = dict(zip(unique_ids, range(len(unique_ids))))
            labels = [ids2labels[id] for id in ids]
        else:
            labels = ids
        samples = [{'im_path': im_path, 'label': label, 'cam': cam} for im_path, label, cam in zip(im_paths, labels, cams)]
        save_pickle(samples, save_path)

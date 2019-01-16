import os.path as osp
from ..dataset import Dataset
from ...utils.file import get_files_by_pattern
from ...utils.file import save_pickle


class PartialiLIDs(Dataset):
    # Will be available in the future
    has_pap_mask = False
    has_ps_label = False
    im_root = 'Partial_iLIDS'
    split_spec = {
            'query': {'pattern': '{}/Probe/*.jpg'.format(im_root), 'map_label': False},
            'gallery': {'pattern': '{}/Gallery/*.jpg'.format(im_root), 'map_label': False},
    }

    def _get_kpt_key(self, im_path):
        return im_path

    def _get_ps_label_path(self, im_path):
        return osp.join(self.root, im_path.replace(self.im_root, self.im_root + '_ps_label').replace('.jpg', '.png'))

    @staticmethod
    def parse_im_path(im_path):
        im_name = osp.basename(im_path)
        id = int(osp.splitext(im_name)[0])
        return id

    def save_split(self, spec, save_path):
        cfg = self.cfg
        im_paths = sorted(get_files_by_pattern(self.root, pattern=spec['pattern'], strip_root=True))
        assert len(im_paths) > 0, "There are {} images for split [{}] of dataset [{}]. Please place your dataset in right position." \
            .format(len(im_paths), cfg.split, self.__class__.__name__)
        # The dataset does not annotate camera. Here we manually set query camera to 0, gallery to 1, to satisfy the testing code.
        ids = [self.parse_im_path(p) for p in im_paths]
        cams = [0 if cfg.split == 'query' else 1 for _ in im_paths]
        if spec['map_label']:
            unique_ids = sorted(list(set(ids)))
            ids2labels = dict(zip(unique_ids, range(len(unique_ids))))
            labels = [ids2labels[id] for id in ids]
        else:
            labels = ids
        samples = [{'im_path': im_path, 'label': label, 'cam': cam} for im_path, label, cam in zip(im_paths, labels, cams)]
        save_pickle(samples, save_path)

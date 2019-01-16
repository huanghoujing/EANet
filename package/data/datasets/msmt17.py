import os.path as osp
from ..dataset import Dataset
from ...utils.file import load_pickle, save_pickle
from ...utils.file import read_lines


class MSMT17(Dataset):
    has_pap_mask = True
    has_ps_label = True
    im_root = 'MSMT17_V1'

    def _get_im_paths(self, list_file, dir_name):
        lines = read_lines(osp.join(self.root, self.im_root, list_file))
        im_paths = sorted([osp.join(self.im_root, dir_name, l.split(' ')[0]) for l in lines])
        return im_paths

    def _gen_split_spec(self):
        split_spec = {
            'train': {'im_paths': self._get_im_paths('list_train.txt', 'train') + self._get_im_paths('list_val.txt', 'train'), 'map_label': True},
            'query': {'im_paths': self._get_im_paths('list_query.txt', 'test'), 'map_label': False},
            'gallery': {'im_paths': self._get_im_paths('list_gallery.txt', 'test'), 'map_label': False},
        }
        return split_spec

    def _get_kpt_key(self, im_path):
        return im_path

    def _get_ps_label_path(self, im_path):
        return osp.join(self.root, im_path.replace(self.im_root, self.im_root + '_ps_label').replace('.jpg', '.png'))

    @staticmethod
    def parse_im_path(im_path):
        im_name = osp.basename(im_path)
        id = int(im_name[:4])
        cam = int(im_name[9:11])
        return id, cam

    def save_split(self, spec, save_path):
        cfg = self.cfg
        im_paths = spec['im_paths']
        assert len(im_paths) > 0, "There are {} images for split [{}] of dataset [{}]. Please place your dataset in right position." \
            .format(len(im_paths), cfg.split, self.__class__.__name__)
        ids, cams = zip(*[self.parse_im_path(p) for p in im_paths])
        if spec['map_label']:
            unique_ids = sorted(list(set(ids)))
            ids2labels = dict(zip(unique_ids, range(len(unique_ids))))
            labels = [ids2labels[id] for id in ids]
        else:
            labels = ids
        samples = [{'im_path': im_path, 'label': label, 'cam': cam} for im_path, label, cam in zip(im_paths, labels, cams)]
        save_pickle(samples, save_path)

    def load_split(self):
        cfg = self.cfg
        self.split_spec = self._gen_split_spec()
        save_path = osp.join(self.root, cfg.split + '.pkl')
        self.save_split(self.split_spec[cfg.split], save_path)
        samples = load_pickle(save_path)
        return samples

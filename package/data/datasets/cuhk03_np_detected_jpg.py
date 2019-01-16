import os.path as osp
from PIL import Image
import numpy as np
from scipy.misc import imsave
from tqdm import tqdm
import shutil
from ...utils.file import walkdir
from ...utils.file import may_make_dir
from .market1501 import Market1501


class CUHK03NpDetectedJpg(Market1501):
    has_pap_mask = True
    has_ps_label = True
    png_im_root = 'cuhk03-np'
    im_root = 'cuhk03-np-jpg'
    split_spec = {
        'train': {'pattern': '{}/detected/bounding_box_train/*.jpg'.format(im_root), 'map_label': True},
        'query': {'pattern': '{}/detected/query/*.jpg'.format(im_root), 'map_label': False},
        'gallery': {'pattern': '{}/detected/bounding_box_test/*.jpg'.format(im_root), 'map_label': False},
    }

    def __init__(self, cfg, samples=None):
        self.root = osp.join(cfg.root, cfg.name)
        self._save_png_as_jpg()
        super(CUHK03NpDetectedJpg, self).__init__(cfg, samples=samples)

    def _get_kpt_key(self, im_path):
        return im_path

    def _get_ps_label_path(self, im_path):
        return osp.join(self.root, im_path.replace(self.im_root, self.im_root + '_ps_label').replace('.jpg', '.png'))

    def _save_png_as_jpg(self):
        png_im_dir = osp.join(self.root, self.png_im_root)
        jpg_im_dir = osp.join(self.root, self.im_root)
        assert osp.exists(png_im_dir), "The PNG image dir {} should be placed inside {}".format(png_im_dir, self.root)
        png_paths = list(walkdir(png_im_dir, '.png'))
        if osp.exists(jpg_im_dir):
            jpg_paths = list(walkdir(jpg_im_dir, '.jpg'))
            if len(png_paths) == len(jpg_paths):
                print('=> Found same number of JPG images. Skip transforming PNG to JPG.')
                return
            else:
                shutil.rmtree(jpg_im_dir)
                print('=> JPG image dir exists but does not contain same number of images. So it is removed and would be re-generated.')
        # CUHK03 contains 14097 detected + 14096 labeled images = 28193
        for png_path in tqdm(png_paths, desc='PNG->JPG', miniters=2000, ncols=120, unit=' images'):
            jpg_path = png_path.replace(self.png_im_root, self.im_root).replace('.png', '.jpg')
            may_make_dir(osp.dirname(jpg_path))
            imsave(jpg_path, np.array(Image.open(png_path)))

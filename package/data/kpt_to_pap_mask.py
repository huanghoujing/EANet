from __future__ import print_function
import numpy as np
from .datasets.coco import kpt_name_to_ind

"""
Some naming shortcuts:
    kpt: keypoint

Visibility:
    0: invisible
    1: visible
"""


def fuse_y(kpts, l_name, r_name, fuse=np.mean):
    l_kpt = kpts[kpt_name_to_ind[l_name]]
    r_kpt = kpts[kpt_name_to_ind[r_name]]
    if (l_kpt[2] == 1) and (r_kpt[2] == 1):
        y = fuse([l_kpt[1], r_kpt[1]])
        visible = True
    elif (l_kpt[2] == 0) and (r_kpt[2] == 1):
        y = r_kpt[1]
        visible = True
    elif (l_kpt[2] == 1) and (r_kpt[2] == 0):
        y = l_kpt[1]
        visible = True
    else:
        y = 0
        visible = False
    return y, visible


PARTS_DICT = {
    'PAP_6P': ('HEAD', 'UPPER_TORSO', 'LOWER_TORSO', 'UPPER_LEG', 'LOWER_LEG', 'SHOES'),
    'PAP_9P': ('HEAD', 'UPPER_TORSO', 'LOWER_TORSO', 'UPPER_LEG', 'LOWER_LEG', 'SHOES', 'UPPER_HALF', 'LOWER_HALF', 'WHOLE'),
}
# When training with random part occlusion, don't occlude these parts.
PARTS_DONT_DROP_DICT = {
    'PAP_6P': [],
    'PAP_9P': ['WHOLE', ],
}


def gen_pap_masks(im_h_w, h_w, kpts, mask_type='PAP_9P'):
    """Generate pap masks for one image.
    Args:
        im_h_w: size of image
        h_w: size of pap masks
        kpts: np array or a list with shape [num_kpts, 3], kpts[i] is (x, y, visibility)
    Returns:
        masks: numpy array (float32) with shape [num_masks, h, w]
        visible: numpy array (float32) with shape [num_masks]. If some keypoints are invisible, related parts may be unavailable.
    """
    parts = PARTS_DICT[mask_type]
    kpts = np.array(kpts)
    assert len(kpts.shape) == 2
    assert kpts.shape[1] == 3
    P = kpts.shape[0]
    H, W = h_w
    masks = []
    visible = []

    def _to_ind(y):
        return min(H, max(0, int(1. * H * y / im_h_w[0] + 0.5)))

    def _to_ind_x(x):
        return min(W, max(0, int(1. * W * x / im_h_w[1] + 0.5)))

    def _gen_mask(y1, y2, v, part_name='', debug=False):
        if debug and v and y1 >= y2:
            print('[Warning][{:15}] y1 {:2d}, y2 {:2d}, v {}'.format(part_name, y1, y2, v))
        v = v and (y1 < y2)
        m = np.zeros([H, W], dtype=np.float32)
        # Return 0 mask if invisible
        if v:
            m[y1:y2] = 1
        masks.append(m)
        visible.append(v)
        return m, v

    # Exclude upper background
    def _gen_HEAD_mask():
        return _gen_mask(max(0, shoulder_y - int(np.ceil(H * 0.25))), shoulder_y, shoulder_v, part_name='HEAD')

    def _gen_UPPER_TORSO_mask():
        return _gen_mask(shoulder_y, shoulder_hip_mid_y, shoulder_hip_mid_v, part_name='UPPER_TORSO')

    def _gen_LOWER_TORSO_mask():
        return _gen_mask(shoulder_hip_mid_y, hip_y, shoulder_hip_mid_v, part_name='LOWER_TORSO')

    def _gen_UPPER_LEG_mask():
        return _gen_mask(hip_y, knee_y, hip_v and knee_v, part_name='UPPER_LEG')

    # Here LOWER_LEG is without shoes
    def _gen_LOWER_LEG_mask():
        if knee_v and ankle_v:
            return _gen_mask(knee_y, ankle_y, True, part_name='LOWER_LEG')
        elif knee_v:
            return _gen_mask(knee_y, min(H, knee_y + int(np.ceil(H * 0.25))), True, part_name='LOWER_LEG')
        else:
            return _gen_mask(0, H, False, part_name='LOWER_LEG')

    def _gen_circle_mask(h_w, pt, radius):
        """A circle region.
        pt: head center point
        """
        h, w = h_w
        x, y = pt
        xv, yv = np.arange(w)[np.newaxis, :], np.arange(h)[:, np.newaxis]
        return ((xv - x) ** 2 + (yv - y) ** 2 <= radius ** 2).astype(np.float32)

    def _gen_SHOES_mask():
        radius = int(0.5 * H / 6)
        left_ankle_kpt = kpts[kpt_name_to_ind['left_ankle']]
        left_shoes_mask = _gen_circle_mask((H, W), (_to_ind_x(left_ankle_kpt[0]), _to_ind(left_ankle_kpt[1])), radius) if left_ankle_kpt[2] == 1 else np.zeros([H, W], dtype=np.float32)
        right_ankle_kpt = kpts[kpt_name_to_ind['right_ankle']]
        right_shoes_mask = _gen_circle_mask((H, W), (_to_ind_x(right_ankle_kpt[0]), _to_ind(right_ankle_kpt[1])), radius) if right_ankle_kpt[2] == 1 else np.zeros([H, W], dtype=np.float32)
        shoes_mask = np.logical_or(left_shoes_mask, right_shoes_mask).astype(np.float32)
        masks.append(shoes_mask)
        visible.append(ankle_v)
        return shoes_mask, ankle_v

    def _gen_UPPER_HALF_mask():
        return _gen_mask(0, hip_y, hip_v, part_name='UPPER_HALF')

    # If knee and ankle are both invisible, view lower half as invisible
    def _gen_LOWER_HALF_mask():
        return _gen_mask(hip_y, H, hip_v and (knee_v or ankle_v), part_name='LOWER_HALF')

    def _gen_WHOLE_mask():
        return _gen_mask(0, H, True, part_name='WHOLE')

    shoulder_y, shoulder_v = fuse_y(kpts, 'left_shoulder', 'right_shoulder')
    hip_y, hip_v = fuse_y(kpts, 'left_hip', 'right_hip')
    knee_y, knee_v = fuse_y(kpts, 'left_knee', 'right_knee')
    ankle_y, ankle_v = fuse_y(kpts, 'left_ankle', 'right_ankle')
    shoulder_hip_mid_y, shoulder_hip_mid_v = (shoulder_y + hip_y) * 0.5, shoulder_v and hip_v
    shoulder_y, hip_y, knee_y, shoulder_hip_mid_y, ankle_y = _to_ind(shoulder_y), _to_ind(hip_y), _to_ind(knee_y), _to_ind(shoulder_hip_mid_y), _to_ind(ankle_y)
    for p in parts:
        eval('_gen_{}_mask()'.format(p))
    masks = np.array(masks, dtype=np.float32)
    visible = np.array(visible, dtype=np.float32)
    return masks, visible

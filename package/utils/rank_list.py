import numpy as np
from .image import read_im, save_im, make_im_grid


def add_border(im, border_width, value):
    """Add color border around an image. The resulting image size is not changed.
    Args:
        im: numpy array with shape [3, im_h, im_w]
        border_width: scalar, measured in pixel
        value: scalar, or numpy array with shape [3]; the color of the border
    Returns:
        im: numpy array with shape [3, im_h, im_w]
    """
    assert (im.ndim == 3) and (im.shape[0] == 3)
    im = np.copy(im)

    if isinstance(value, np.ndarray):
        # reshape to [3, 1, 1]
        value = value.flatten()[:, np.newaxis, np.newaxis]
    im[:, :border_width, :] = value
    im[:, -border_width:, :] = value
    im[:, :, :border_width] = value
    im[:, :, -border_width:] = value

    return im


def get_rank_list(dist_vec, q_id=None, q_cam=None, g_ids=None, g_cams=None,
                  rank_list_size=10, skip_same_id_same_cam=True, id_aware=False):
    """Get the ranking list of a query image
    Args:
        dist_vec: a numpy array with shape [num_gallery_images], the distance between the query image and all gallery images
        q_id: optional, a scalar, query id
        q_cam: optional, a scalar, query camera
        g_ids: optional, a numpy array with shape [num_gallery_images], gallery ids
        g_cams: optional, a numpy array with shape [num_gallery_images], gallery cameras
        rank_list_size: a scalar, the number of images to show in a rank list
        skip_same_id_same_cam: Skip gallery images with same id and same camera as query
        id_aware: whether q_id, q_cam, g_ids, g_cams are known
    Returns:
        rank_list: a list, the indices of gallery images to show
        same_id: None, or a list, len(same_id) = rank_list, whether each ranked image is with same id as query
    """
    sort_inds = np.argsort(dist_vec)

    if not id_aware:
        rank_list = sort_inds[:rank_list_size]
        return rank_list, None

    assert q_id is not None
    assert q_cam is not None
    assert g_ids is not None
    assert g_cams is not None

    rank_list = []
    same_id = []
    i = 0
    for ind in sort_inds:
        g_id, g_cam = g_ids[ind], g_cams[ind]
        # Skip gallery images with same id and same camera as query
        if (q_id == g_id) and (q_cam == g_cam) and skip_same_id_same_cam:
            continue
        same_id.append(q_id == g_id)
        rank_list.append(ind)
        i += 1
        if i >= rank_list_size:
            break
    return rank_list, same_id


def save_rank_list_to_im(rank_list, q_im_path, g_im_paths, save_path, same_id=None, resize_h_w=(128, 64)):
    """Save a query and its rank list as an image.
    Args:
        rank_list: a list, the indices of gallery images to show
        q_im_path: query image path
        g_im_paths: ALL gallery image paths
        save_path: path to save the query and its rank list as an image
        same_id: optional, a list, len(same_id) = rank_list, whether each ranked image is with same id as query
        resize_h_w: size of each image in the rank list
    """
    ims = [read_im(q_im_path, convert_rgb=True, resize_h_w=resize_h_w, transpose=True)]
    for i, ind in enumerate(rank_list):
        im = read_im(g_im_paths[ind], convert_rgb=True, resize_h_w=resize_h_w, transpose=True)
        if same_id is not None:
            # Add green boundary to true positive, red to false positive
            color = np.array([0, 255, 0]) if same_id[i] else np.array([255, 0, 0])
            im = add_border(im, 3, color)
        ims.append(im)
    im = make_im_grid(ims, 1, len(rank_list) + 1, 8, 255)
    save_im(im, save_path, transpose=True)

import os
import os.path as osp
import shutil
import json
import numpy as np
import glob
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def may_make_dir(path):
    """
    Args:
        path: a dir, e.g. result of `osp.dirname()`
    Note:
        `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
    """
    # This clause has mistakes:
    # if path is None or '':

    if path in [None, '']:
        return
    if not osp.exists(path):
        os.makedirs(path)


def load_pickle(path, verbose=True):
    """Check and load pickle object.
    According to this post: https://stackoverflow.com/a/41733927, cPickle and
    disabling garbage collector helps with loading speed."""
    assert osp.exists(path), "File not exists: {}".format(path)
    # gc.disable()
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    # gc.enable()
    if verbose:
        print('Loaded pickle file {}'.format(path))
    return ret


def save_pickle(obj, path, verbose=True):
    """Create dir and save file."""
    may_make_dir(osp.dirname(osp.abspath(path)))
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)
    if verbose:
        print('Pickle file saved to {}'.format(path))


def load_json(path):
    """Check and load json file."""
    assert osp.exists(path), "Json file not exists: {}".format(path)
    with open(path, 'r') as f:
        ret = json.load(f)
    print('Loaded json file {}'.format(path))
    return ret


def save_json(obj, path):
    """Create dir and save file."""
    may_make_dir(osp.dirname(osp.abspath(path)))
    with open(path, 'w') as f:
        json.dump(obj, f)
    print('Json file saved to {}'.format(path))


def read_lines(file):
    with open(file) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines if l.strip()]
    return lines


def copy_to(p1, p2):
    # Only when the copy can go on without error do we create destination dir.
    if osp.exists(p1):
        may_make_dir(osp.dirname(p2))
    shutil.copy(p1, p2)


def get_files_by_pattern(root, pattern='a/b/*.ext', strip_root=False):
    """Optionally to only return matched sub paths."""
    ret = glob.glob(osp.join(root, pattern))
    if strip_root:
        ret = [r[len(root) + 1:] for r in ret]
    return ret


def walkdir(folder, exts=None, sub_path=False, abs_path=False):
    """Walk through each files in a directory.
    Reference: https://github.com/tqdm/tqdm/wiki/How-to-make-a-great-Progress-Bar
    Args:
        exts: file extensions, e.g. '.jpg', or ['.jpg'] or ['.jpg', '.png']
        sub_path: whether to exclude `folder` in the resulting paths, remaining sub paths
        abs_path: whether to return absolute paths
    """
    if isinstance(exts, str):
        exts = [exts]
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            if (exts is None) or (os.path.splitext(filename)[1] in exts):
                path = os.path.join(dirpath, filename)
                if sub_path:
                    path = path[len(folder) + 1:]
                elif abs_path:
                    path = os.path.abspath(path)
                yield path


def strip_root(path):
    """a/b/c -> b/c"""
    sep = os.sep
    path = sep.join(path.split(sep)[1:])
    return path

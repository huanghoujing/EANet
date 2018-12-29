from __future__ import print_function
import sys
import os
from os.path import abspath as ospap
from os.path import dirname as ospdn


def get_conda_paths():
    return [p for p in sys.path if ('conda' in p)]


def get_non_conda_paths():
    return [p for p in sys.path if ('conda' not in p)]


def mv_conda_paths_to_front():
    """When using conda, we move all conda package paths to the front.
    Otherwise, some paths (e.g. $HOME/.local/python*) may be interrupting."""
    sys.path = get_conda_paths() + get_non_conda_paths()


def insert_package_path():
    """This init_path.py should be placed at `${PROJECT_DIR}/package/utils/init_path.py`.
    Here we add `${PROJECT_DIR}/package` to `sys.path`.
    If you want to move this file to other places, remember to change the following path to insert.
    """
    sys.path.insert(0, ospdn(ospdn(ospdn(ospap(__file__)))))


def init_path():
    print('[PYTHONPATH]:\n\t{}'.format(os.environ.get('PYTHONPATH', '')))
    print('[Original sys.path]:\n\t{}'.format('\n\t'.join(sys.path)))
    mv_conda_paths_to_front()
    insert_package_path()
    print('[Final sys.path]:\n\t{}'.format('\n\t'.join(sys.path)))
    # import torch
    # print('[PYTORCH VERSION]:', torch.__version__)


init_path()
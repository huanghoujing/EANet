"""Remove optimizer and lr scheduler in checkpoint to reduce file size."""
from __future__ import print_function


import sys
sys.path.insert(0, '.')
from package.utils.file import get_files_by_pattern
from package.utils.torch_utils import only_keep_model


ori_ckpt_paths = get_files_by_pattern('exp/eanet/test_paper_models', '*/*/*.pth')
new_ckpt_paths = ori_ckpt_paths

for ori_path, new_path in zip(ori_ckpt_paths, new_ckpt_paths):
    only_keep_model(ori_path, new_path)

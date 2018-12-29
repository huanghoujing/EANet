from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """Modified from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py"""
    def __init__(self, data_source, k=1):
        self.data_source = data_source
        self.k = k
        self.index_dic = defaultdict(list)
        for index, sample in enumerate(data_source):
            self.index_dic[sample['label']].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_pids = len(self.pids)

    def __len__(self):
        return self.num_pids * self.k

    def __iter__(self):
        indices = torch.randperm(self.num_pids)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.k:
                t = np.random.choice(t, size=self.k, replace=False)
            else:
                t = np.random.choice(t, size=self.k, replace=True)
            ret.extend(t)
        return iter(ret)

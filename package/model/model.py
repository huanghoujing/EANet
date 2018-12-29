from __future__ import print_function
from itertools import chain
import torch
import torch.nn as nn
from .base_model import BaseModel
from .backbone import create_backbone
from .pa_pool import PAPool
from .pcb_pool import PCBPool
from .global_pool import GlobalPool
from .ps_head import PartSegHead
from ..utils.model import create_embedding
from ..utils.model import init_classifier


class Model(BaseModel):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg.backbone)
        self.pool = eval('{}(cfg)'.format(cfg.pool_type))
        self.create_em_list()
        if hasattr(cfg, 'num_classes') and cfg.num_classes > 0:
            self.create_cls_list()
        if cfg.use_ps:
            cfg.ps_head.in_c = self.backbone.out_c
            self.ps_head = PartSegHead(cfg.ps_head)
        print('Model Structure:\n{}'.format(self))

    def create_em_list(self):
        cfg = self.cfg
        self.em_list = nn.ModuleList([create_embedding(self.backbone.out_c, cfg.em_dim) for _ in range(cfg.num_parts)])

    def create_cls_list(self):
        cfg = self.cfg
        self.cls_list = nn.ModuleList([nn.Linear(cfg.em_dim, cfg.num_classes) for _ in range(cfg.num_parts)])
        ori_w = self.cls_list[0].weight.view(-1).detach().numpy().copy()
        self.cls_list.apply(init_classifier)
        new_w = self.cls_list[0].weight.view(-1).detach().numpy().copy()
        import numpy as np
        if np.array_equal(ori_w, new_w):
            from ..utils.log import array_str
            print('!!!!!! Warning: Model Weight Not Changed After Init !!!!!')
            print('Original Weight [:20]:\n\t{}'.format(array_str(ori_w[:20], fmt='{:.6f}')))
            print('New Weight [:20]:\n\t{}'.format(array_str(new_w[:20], fmt='{:.6f}')))

    def get_ft_and_new_params(self, cft=False):
        """cft: Clustering and Fine Tuning"""
        ft_modules, new_modules = self.get_ft_and_new_modules(cft=cft)
        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules]))
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules]))
        return ft_params, new_params

    def get_ft_and_new_modules(self, cft=False):
        if cft:
            ft_modules = [self.backbone, self.em_list]
            if hasattr(self, 'ps_head'):
                ft_modules += [self.ps_head]
            new_modules = [self.cls_list] if hasattr(self, 'cls_list') else []
        else:
            ft_modules = [self.backbone]
            new_modules = [self.em_list]
            if hasattr(self, 'cls_list'):
                new_modules += [self.cls_list]
            if hasattr(self, 'ps_head'):
                new_modules += [self.ps_head]
        return ft_modules, new_modules

    def set_train_mode(self, cft=False, fix_ft_layers=False):
        self.train()
        if fix_ft_layers:
            for m in self.get_ft_and_new_modules(cft=cft)[0]:
                m.eval()

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict['im'])

    def reid_forward(self, in_dict):
        # print('=====> in_dict.keys() entering reid_forward():', in_dict.keys())
        pool_out_dict = self.pool(in_dict)
        feat_list = [em(f) for em, f in zip(self.em_list, pool_out_dict['feat_list'])]
        out_dict = {
            'feat_list': feat_list,
        }
        if hasattr(self, 'cls_list'):
            logits_list = [cls(f) for cls, f in zip(self.cls_list, feat_list)]
            out_dict['logits_list'] = logits_list
        if 'visible' in pool_out_dict:
            out_dict['visible'] = pool_out_dict['visible']
        return out_dict

    def ps_forward(self, in_dict):
        return self.ps_head(in_dict['feat'])

    def forward(self, in_dict, forward_type='reid'):
        in_dict['feat'] = self.backbone_forward(in_dict)
        if forward_type == 'reid':
            out_dict = self.reid_forward(in_dict)
        elif forward_type == 'ps':
            out_dict = {'ps_pred': self.ps_forward(in_dict)}
        elif forward_type == 'ps_reid_parallel':
            out_dict = self.reid_forward(in_dict)
            out_dict['ps_pred'] = self.ps_forward(in_dict)
        elif forward_type == 'ps_reid_serial':
            ps_pred = self.ps_forward(in_dict)
            # Generate pap masks from ps_pred
            in_dict['pap_mask'] = gen_pap_mask_from_ps_pred(ps_pred)
            out_dict = self.reid_forward(in_dict)
            out_dict['ps_pred'] = ps_pred
        else:
            raise ValueError('Error forward_type {}'.format(forward_type))
        return out_dict

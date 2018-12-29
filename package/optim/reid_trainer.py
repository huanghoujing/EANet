"""Manager for common ReID training and optional testing.
TODO: Separate into a trainer (model, optim, lr) and an evaluator (containing model, eval)
"""
from __future__ import print_function
import argparse
import time
from collections import OrderedDict
import os
import os.path as osp
from copy import deepcopy
from torch.nn.parallel import DataParallel

from ..utils.misc import import_file
from ..utils.file import copy_to
from ..utils.cfg import transfer_items
from ..utils.cfg import overwrite_cfg_file
from ..utils.torch_utils import get_default_device
from ..utils.torch_utils import load_ckpt, save_ckpt
from ..utils.torch_utils import get_optim_lr_str
from ..utils.log import ReDirectSTD
from ..utils.log import time_str as t_str
from ..utils.log import join_str
from ..data.dataloader import create_dataloader
from ..data.create_dataset import dataset_shortcut as d_sc
from ..utils.log import score_str as s_str
from ..utils.log import write_to_file
from .trainer import Trainer
from ..eval.eval_dataloader import eval_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='None', help='[Optional] Directory to store experiment output, including log files and model checkpoint, etc.')
    parser.add_argument('--cfg_file', type=str, default='None', help='A configuration file.')
    parser.add_argument('--ow_file', type=str, default='None', help='[Optional] A text file, each line being an item to overwrite the cfg_file.')
    parser.add_argument('--ow_str', type=str, default='None', help="""[Optional] Items to overwrite the cfg_file. E.g. "cfg.dataset.train.name = 'market1501'; cfg.model.em_dim = 256" """)
    args = parser.parse_args()
    return args


class ReIDTrainer(object):
    """Note: This class does not inherit but contains Trainer."""
    def __init__(self):
        self.init_cfg()
        self.init_log()
        self.init_device()
        if not self.cfg.only_test:
            self.init_trainer()
        self.init_eval()

    def init_cfg(self):
        args = parse_args()
        exp_dir = args.exp_dir
        if exp_dir == 'None':
            exp_dir = 'exp/' + osp.splitext(osp.basename(args.cfg_file))[0]
        # Copy the config file to exp_dir, and then overwrite any configurations provided in ow_file and ow_str
        cfg_file = osp.join(exp_dir, osp.basename(args.cfg_file))
        copy_to(args.cfg_file, cfg_file)
        if args.ow_file != 'None':
            # print('ow_file is: {}'.format(args.ow_file))
            overwrite_cfg_file(cfg_file, ow_file=args.ow_file)
        if args.ow_str != 'None':
            # print('ow_str is: {}'.format(args.ow_str))
            overwrite_cfg_file(cfg_file, ow_str=args.ow_str)
        self.cfg = import_file(cfg_file).cfg
        # Tricky! EasyDict.__setattr__ will transform tuple into list!
        # print('=====> type(cfg.dataset.pap_mask.h_w):', type(self.cfg.dataset.pap_mask.h_w))
        self.cfg.log.exp_dir = exp_dir

    def init_log(self):
        cfg = self.cfg.log
        # Redirect logs to both console and file.
        time_str = t_str()
        ReDirectSTD(osp.join(cfg.exp_dir, 'stdout_{}.txt'.format(time_str)), 'stdout', True)
        ReDirectSTD(osp.join(cfg.exp_dir, 'stderr_{}.txt'.format(time_str)), 'stderr', True)
        print('=> Experiment Output Directory: {}'.format(self.cfg.log.exp_dir))
        import torch
        print('[PYTORCH VERSION]:', torch.__version__)
        cfg.ckpt_file = osp.join(cfg.exp_dir, 'ckpt.pth')
        if cfg.use_tensorboard:
            from tensorboardX import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
        else:
            self.tb_writer = None
        cfg.score_file = osp.join(cfg.exp_dir, 'score_{}.txt'.format(time_str))

    def init_device(self):
        self.device = get_default_device()
        self.cfg.eval.device = self.device

    def init_trainer(self, samples=None):
        cfg = self.cfg
        self.create_train_loader(samples=samples)
        self.create_model()
        self.create_optimizer()
        self.create_lr_scheduler()
        self.create_loss_funcs()
        self.trainer = Trainer(self.train_loader, self.train_forward, self.criterion, self.optimizer, self.lr_scheduler,
                               steps_per_log=cfg.optim.steps_per_log, print_step_log=self.print_log)
        self.ckpt_objects = {'model': self.model, 'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}
        if cfg.optim.resume:
            self.resume()

    def init_eval(self):
        if not hasattr(self, 'model'):
            self.create_model()
        if self.cfg.only_test:
            self.load_items(model=True)
        self.create_test_loaders()

    def load_items(self, model=False, optimizer=False, lr_scheduler=False):
        """To allow flexible multi-stage training."""
        cfg = self.cfg.log
        objects = {}
        if model:
            objects['model'] = self.model
        if optimizer:
            objects['optimizer'] = self.optimizer
        if lr_scheduler:
            objects['lr_scheduler'] = self.lr_scheduler
        load_ckpt(objects, cfg.ckpt_file, strict=False)

    def resume(self):
        """This method is ONLY used for resuming training after program breakdown.
        self.cfg.optim.resume is also ONLY used for this purpose.
        For finetuning or changing training phase, manually call self.load_items(**kwargs)."""
        cfg = self.cfg.log
        resume_ep, score = load_ckpt(self.ckpt_objects, cfg.ckpt_file)
        self.trainer.current_ep = resume_ep
        self.trainer.current_step = resume_ep * len(self.train_loader)

    def create_dataloader(self, mode=None, name=None, split=None, samples=None):
        """Dynamically create any split of any dataset, with dynamic mode."""
        cfg = self.cfg
        assert mode in ['train', 'cd_train', 'test']
        transfer_items(getattr(cfg.dataset, mode), cfg.dataset)
        transfer_items(getattr(cfg.dataloader, mode), cfg.dataloader)
        if name is not None:
            cfg.dataset.name = name
        if split is not None:
            cfg.dataset.split = split
        # NOTE: `transfer_items`, `cfg.dataset.name = name` etc change cfg in place.
        # Deepcopy prevents next call of `self.create_dataloader` from modifying the
        # cfg stored in the previous dataset.
        dataloader = create_dataloader(deepcopy(cfg.dataloader), deepcopy(cfg.dataset), samples=samples)
        return dataloader

    def create_test_loaders(self):
        cfg = self.cfg
        self.test_loaders = OrderedDict()
        for i, name in enumerate(cfg.dataset.test.names):
            q_split = cfg.dataset.test.query_splits[i] if hasattr(cfg.dataset.test, 'query_splits') else 'query'
            self.test_loaders[name] = {
                'query': self.create_dataloader(mode='test', name=name, split=q_split),
                'gallery': self.create_dataloader(mode='test', name=name, split='gallery')
            }

    def test(self):
        cfg = self.cfg
        score_strs = []
        score_summary = []
        for test_name, loader_dict in self.test_loaders.items():
            cfg.eval.test_feat_cache_file = osp.join(cfg.log.exp_dir, '{}_to_{}_feat_cache.pkl'.format(d_sc[cfg.dataset.train.name], d_sc[test_name]))
            cfg.eval.score_prefix = '{} -> {}'.format(d_sc[cfg.dataset.train.name], d_sc[test_name]).ljust(12)
            # Due to an abnormal bug, I decide not to use DataParallel during testing.
            # The bug case: total im 15913, batch size 32, 15913 % 32 = 9, it's ok to use 2 gpus,
            # but when I used 4 gpus, it threw error at the last batch: [line 83, in parallel_apply
            # , ... TypeError: forward() takes at least 2 arguments (2 given)]
            model_for_eval = self.model.module if isinstance(self.model, DataParallel) else self.model
            score_dict = eval_dataloader(model_for_eval, loader_dict['query'], loader_dict['gallery'], deepcopy(cfg.eval))
            score_strs.append(score_dict['scores_str'])
            score_summary.append("{}->{}: {} ({})".format(d_sc[cfg.dataset.train.name], d_sc[test_name], s_str(score_dict['cmc_scores'][0]).replace('%', ''), s_str(score_dict['mAP']).replace('%', '')))
        score_str = join_str(score_strs, '\n')
        score_summary = ('Epoch {}'.format(self.trainer.current_ep) if hasattr(self, 'trainer') else 'Test').ljust(12) + ', '.join(score_summary) + '\n'
        write_to_file(cfg.log.score_file, score_summary, append=True)
        return score_str

    def create_train_loader(self):
        # Create self.train_loader
        raise NotImplementedError

    def create_model(self):
        # Create self.model, then
        #     from package.utils.torch_utils import may_data_parallel
        #     self.model = may_data_parallel(self.model)
        #     self.model.to(self.device)
        raise NotImplementedError

    def set_model_to_train_mode(self):
        # Default is self.model.train()
        raise NotImplementedError

    def create_optimizer(self):
        # Create self.optimizer, then
        #     recursive_to_device(self.optimizer.state_dict(), self.device)
        # self.optimizer.to(self.device) # One day, there may be this function in official pytorch
        raise NotImplementedError

    def create_lr_scheduler(self):
        # Create self.lr_scheduler, self.epochs.
        # self.lr_scheduler can be set to None
        # TODO: a better place to set cfg.epochs?
        raise NotImplementedError

    def create_loss_funcs(self):
        # Create self.loss_funcs, an OrderedDict
        raise NotImplementedError

    def train_forward(self, batch):
        # pred = self.train_forward(batch)
        raise NotImplementedError

    def criterion(self, batch, pred):
        # loss = self.criterion(batch, pred)
        raise NotImplementedError

    def get_log(self):
        time_log = 'Ep {}, Step {}, {:.2f}s'.format(self.trainer.current_ep + 1, self.trainer.current_step + 1, time.time() - self.ep_st)
        lr_log = 'lr {}'.format(get_optim_lr_str(self.optimizer))
        meter_log = join_str([m.avg_str for lf in self.loss_funcs.values() for m in lf.meter_dict.values()], ', ')
        log = join_str([time_log, lr_log, meter_log], ', ')
        return log

    def print_log(self):
        print(self.get_log())

    def may_test(self):
        cfg = self.cfg.optim
        score_str = ''
        # You can force not testing by manually setting dont_test=True.
        if not hasattr(cfg, 'dont_test') or not cfg.dont_test:
            if (self.trainer.current_ep % cfg.epochs_per_val == 0) or (self.trainer.current_ep == cfg.epochs) or cfg.trial_run:
                score_str = self.test()
        return score_str

    def may_save_ckpt(self, score_str):
        cfg = self.cfg
        if not cfg.optim.trial_run:
            save_ckpt(self.ckpt_objects, self.trainer.current_ep, score_str, cfg.log.ckpt_file)

    def train(self):
        cfg = self.cfg.optim
        for _ in range(self.trainer.current_ep, cfg.epochs):
            self.ep_st = time.time()
            self.set_model_to_train_mode()
            self.trainer.train_one_epoch(trial_run_steps=3 if cfg.trial_run else None)
            score_str = self.may_test()
            self.may_save_ckpt(score_str)

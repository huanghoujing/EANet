from __future__ import print_function
from collections import OrderedDict
from copy import deepcopy
import torch
from torch.optim.lr_scheduler import MultiStepLR
from .reid_trainer import ReIDTrainer
from ..model.model import Model
from ..utils.torch_utils import may_data_parallel
from ..utils.torch_utils import recursive_to_device
from .optimizer import create_optimizer
from .lr_scheduler import WarmupLR
from ..data.multitask_dataloader import MTDataLoader
from ..loss.triplet_loss import TripletLoss
from ..loss.id_loss import IDLoss
from ..loss.ps_loss import PSLoss


class EANetTrainer(ReIDTrainer):

    def create_train_loader(self, samples=None):
        cfg = self.cfg
        self.train_loader = self.create_dataloader(mode='train', samples=samples)
        if cfg.cd_ps_loss.use:
            self.cd_train_loader = self.create_dataloader(mode='cd_train', samples=samples)
            self.train_loader = MTDataLoader([self.train_loader, self.cd_train_loader], ref_loader_idx=0)

    def create_model(self):
        if hasattr(self, 'train_loader'):
            reid_loader = self.train_loader.loaders[0] if self.cfg.cd_ps_loss.use else self.train_loader
            self.cfg.model.num_classes = reid_loader.dataset.num_ids
        self.model = Model(deepcopy(self.cfg.model))
        self.model = may_data_parallel(self.model)
        self.model.to(self.device)

    def set_model_to_train_mode(self):
        cfg = self.cfg.optim
        self.model.set_train_mode(cft=cfg.cft, fix_ft_layers=cfg.phase == 'pretrain')

    def create_optimizer(self):
        cfg = self.cfg.optim
        ft_params, new_params = self.model.get_ft_and_new_params(cft=cfg.cft)
        if cfg.phase == 'pretrain':
            assert len(new_params) > 0, "No new params to pretrain!"
            param_groups = [{'params': new_params, 'lr': cfg.new_params_lr}]
        else:
            param_groups = [{'params': ft_params, 'lr': cfg.ft_lr}]
            # Some model may not have new params
            if len(new_params) > 0:
                param_groups += [{'params': new_params, 'lr': cfg.new_params_lr}]
        self.optimizer = create_optimizer(param_groups, cfg)
        recursive_to_device(self.optimizer.state_dict(), self.device)

    def create_lr_scheduler(self):
        cfg = self.cfg.optim
        if cfg.phase == 'normal':
            cfg.lr_decay_steps = [len(self.train_loader) * ep for ep in cfg.lr_decay_epochs]
            cfg.epochs = cfg.normal_epochs
            self.lr_scheduler = MultiStepLR(self.optimizer, cfg.lr_decay_steps)
        elif cfg.phase == 'warmup':
            cfg.warmup_steps = cfg.warmup_epochs * len(self.train_loader)
            cfg.epochs = cfg.warmup_epochs
            self.lr_scheduler = WarmupLR(self.optimizer, cfg.warmup_steps)
        elif cfg.phase == 'pretrain':
            cfg.pretrain_new_params_steps = cfg.pretrain_new_params_epochs * len(self.train_loader)
            cfg.epochs = cfg.pretrain_new_params_epochs
            self.lr_scheduler = None
        else:
            raise ValueError('Invalid phase {}'.format(cfg.phase))

    def create_loss_funcs(self):
        cfg = self.cfg
        self.loss_funcs = OrderedDict()
        if cfg.id_loss.use:
            self.loss_funcs[cfg.id_loss.name] = IDLoss(cfg.id_loss, self.tb_writer)
        if cfg.tri_loss.use:
            self.loss_funcs[cfg.tri_loss.name] = TripletLoss(cfg.tri_loss, self.tb_writer)
        if cfg.src_ps_loss.use:
            self.loss_funcs[cfg.src_ps_loss.name] = PSLoss(cfg.src_ps_loss, self.tb_writer)
        if cfg.cd_ps_loss.use:
            self.loss_funcs[cfg.cd_ps_loss.name] = PSLoss(cfg.cd_ps_loss, self.tb_writer)

    # NOTE: To save GPU memory, our multi-domain training requires
    # [1st batch: source-domain forward and backward]-
    # [2nd batch: cross-domain forward and backward]-
    # [update model]
    # So the following three-step framework is not strictly followed.
    #     pred = self.train_forward(batch)
    #     loss = self.criterion(batch, pred)
    #     loss.backward()
    def train_forward(self, batch):
        cfg = self.cfg
        batch = recursive_to_device(batch, self.device)
        if cfg.cd_ps_loss.use:
            reid_batch, cd_ps_batch = batch
        else:
            reid_batch = batch
        # Source Loss
        loss = 0
        pred = self.model.forward(reid_batch, forward_type='ps_reid_parallel' if cfg.src_ps_loss.use else 'reid')
        for loss_cfg in [cfg.id_loss, cfg.tri_loss, cfg.src_ps_loss]:
            if loss_cfg.use:
                loss += self.loss_funcs[loss_cfg.name](reid_batch, pred, step=self.trainer.current_step)['loss']
        if isinstance(loss, torch.Tensor):
            loss.backward()
        # Cross-Domain Loss
        if cfg.cd_ps_loss.use:
            pred = self.model.forward(cd_ps_batch, forward_type='ps')
            loss = self.loss_funcs[cfg.cd_ps_loss.name](cd_ps_batch, pred, step=self.trainer.current_step)['loss']
            if isinstance(loss, torch.Tensor):
                loss.backward()

    def criterion(self, batch, pred):
        return 0

    def train_phases(self):
        cfg = self.cfg.optim
        if cfg.warmup:
            cfg.phase = 'warmup'
            cfg.dont_test = True
            self.init_trainer()
            self.train()
            cfg.phase = 'normal'
            cfg.dont_test = False
            self.init_trainer()
            self.load_items(model=True, optimizer=True)
        self.train()


if __name__ == '__main__':
    from ..utils import init_path
    trainer = EANetTrainer()
    if trainer.cfg.only_test:
        trainer.test()
    else:
        trainer.train_phases()

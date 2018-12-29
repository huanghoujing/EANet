from __future__ import print_function
import torch
from .loss import Loss
from ..utils.meter import RecentAverageMeter as Meter


class PSLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(PSLoss, self).__init__(cfg, tb_writer=tb_writer)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none' if cfg.normalize_size else 'mean')

    # TODO: Pytorch newer versions support high-dimension CrossEntropyLoss, so no need to reshape pred and label.
    def __call__(self, batch, pred, step=0, **kwargs):
        cfg = self.cfg

        # Calculation
        ps_pred = pred['ps_pred']
        ps_label = batch['ps_label']
        N, C, H, W = ps_pred.size()
        assert ps_label.size() == (N, H, W)
        # shape [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        ps_pred = ps_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # shape [N, H, W] -> [NHW]
        ps_label = ps_label.view(N * H * W).detach()
        loss = self.criterion(ps_pred, ps_label)
        # Calculate each class avg loss and then average across classes, to compensate for classes that have few pixels
        if cfg.normalize_size:
            loss_ = 0
            cur_batch_n_classes = 0
            for i in range(cfg.num_classes):
                loss_i = loss[ps_label == i]
                if loss_i.numel() > 0:
                    loss_ += loss_i.mean()
                    cur_batch_n_classes += 1
            loss_ /= (cur_batch_n_classes + 1e-8)
            loss = loss_

        # Meter
        if cfg.name not in self.meter_dict:
            self.meter_dict[cfg.name] = Meter(name=cfg.name)
        self.meter_dict[cfg.name].update(loss.item())

        # Tensorboard
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(cfg.name, {cfg.name: self.meter_dict[cfg.name].avg}, step)

        # Scale by loss weight
        loss *= cfg.weight

        return {'loss': loss}

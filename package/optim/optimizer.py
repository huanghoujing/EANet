import torch.optim as optim


def create_optimizer(param_groups, cfg):
    if cfg.optimizer == 'sgd':
        optim_class = optim.SGD
    elif cfg.optimizer == 'adam':
        optim_class = optim.Adam
    else:
        raise NotImplementedError('Unsupported Optimizer {}'.format(cfg.optimizer))
    optim_kwargs = dict(weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'sgd':
        optim_kwargs['momentum'] = cfg.sgd.momentum
        optim_kwargs['nesterov'] = cfg.sgd.nesterov
    return optim_class(param_groups, **optim_kwargs)

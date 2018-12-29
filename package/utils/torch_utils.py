from __future__ import print_function
import os.path as osp
import torch
from torch.nn.parallel import DataParallel
from .file import may_make_dir


def get_default_device():
    """Get default device for `*.to(device)`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def recursive_to_device(input, device):
    """NOTE: If input is dict/list/tuple, it is changed in place."""
    if isinstance(input, torch.Tensor):
        # print('=> IS torch.Tensor')
        # print('=> input.device before to_device: {}'.format(input.device))
        input = input.to(device)
        # print('=> input.device after to_device: {}'.format(input.device))
    elif isinstance(input, dict):
        for k, v in input.items():
            input[k] = recursive_to_device(v, device)
    elif isinstance(input, (list, tuple)):
        input = [recursive_to_device(v, device) for v in input]
    return input


class TransparentDataParallel(DataParallel):

    def __getattr__(self, name):
        """Forward attribute access to its wrapped module."""
        try:
            return super(TransparentDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self, *args, **kwargs):
        """We only save/load state_dict of the wrapped model. This allows loading
        state_dict of a DataParallelSD model into a non-DataParallel model."""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)


def may_data_parallel(model):
    """When there is no more than one gpu, don't wrap the model, for more
    flexibility in forward function."""
    if torch.cuda.device_count() > 1:
        model = TransparentDataParallel(model)
    return model


def load_state_dict(model, src_state_dict, fold_bnt=True):
    """Copy parameters and buffers from `src_state_dict` into `model` and its
    descendants. The `src_state_dict.keys()` NEED NOT exactly match
    `model.state_dict().keys()`. For dict key mismatch, just
    skip it; for copying error, just output warnings and proceed.

    Arguments:
        model: A torch.nn.Module object.
        src_state_dict (dict): A dict containing parameters and persistent buffers.
    Note:
        This is modified from torch.nn.modules.module.load_state_dict(), to make
        the warnings and errors more detailed.
    """
    from torch.nn import Parameter

    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        except Exception, msg:
            print("Warning: Error occurs when copying '{}': {}".format(name, str(msg)))

    # New version of BN has buffer `num_batches_tracked`, which is not used
    # for normal BN, so we fold all these missing keys into one line
    def _fold_nbt(keys):
        nbt_keys = [s for s in keys if s.endswith('.num_batches_tracked')]
        if len(nbt_keys) > 0:
            keys = [s for s in keys if not s.endswith('.num_batches_tracked')] + ['num_batches_tracked  x{}'.format(len(nbt_keys))]
        return keys

    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("Keys not found in source state_dict: ")
        if fold_bnt:
            src_missing = _fold_nbt(src_missing)
        for n in src_missing:
            print('\t', n)

    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        print("Keys not found in destination state_dict: ")
        if fold_bnt:
            dest_missing = _fold_nbt(dest_missing)
        for n in dest_missing:
            print('\t', n)


def load_ckpt(objects, ckpt_file, strict=True):
    """Load state_dict's of modules/optimizers/lr_schedulers from file.
    Args:
        objects: A dict, which values are either
            torch.nn.optimizer
            or torch.nn.Module
            or torch.optim.lr_scheduler._LRScheduler
            or None
        ckpt_file: The file path.
    """
    assert osp.exists(ckpt_file), "ckpt_file {} does not exist!".format(ckpt_file)
    assert osp.isfile(ckpt_file), "ckpt_file {} is not file!".format(ckpt_file)
    ckpt = torch.load(ckpt_file, map_location=(lambda storage, loc: storage))
    for name, obj in objects.items():
        if obj is not None:
            # Only nn.Module.load_state_dict has this keyword argument
            if not isinstance(obj, torch.nn.Module) or strict:
                obj.load_state_dict(ckpt['state_dicts'][name])
            else:
                load_state_dict(obj, ckpt['state_dicts'][name])
    objects_str = ', '.join(objects.keys())
    msg = '=> Loaded [{}] from {}, epoch {}, score:\n{}'.format(objects_str, ckpt_file, ckpt['epoch'], ckpt['score'])
    print(msg)
    return ckpt['epoch'], ckpt['score']


def save_ckpt(objects, epoch, score, ckpt_file):
    """Save state_dict's of modules/optimizers/lr_schedulers to file.
    Args:
        objects: A dict, which members are either
            torch.nn.optimizer
            or torch.nn.Module
            or torch.optim.lr_scheduler._LRScheduler
            or None
        epoch: the current epoch number
        score: the performance of current model
        ckpt_file: The file path.
    Note:
        torch.save() reserves device type and id of tensors to save, so when
        loading ckpt, you have to inform torch.load() to load these tensors to
        cpu or your desired gpu, if you change devices.
    """
    state_dicts = {name: obj.state_dict() for name, obj in objects.items() if obj is not None}
    ckpt = dict(state_dicts=state_dicts,
                epoch=epoch,
                score=score)
    may_make_dir(osp.dirname(ckpt_file))
    torch.save(ckpt, ckpt_file)
    msg = '=> Checkpoint Saved to {}'.format(ckpt_file)
    print(msg)


def only_keep_model(ori_ckpt_path, new_ckpt_path):
    """Remove optimizer and lr scheduler in the checkpoint, reducing file size."""
    ckpt = torch.load(ori_ckpt_path, map_location=(lambda storage, loc: storage))
    ckpt['state_dicts'] = {'model': ckpt['state_dicts']['model']}
    may_make_dir(osp.dirname(new_ckpt_path))
    torch.save(ckpt, new_ckpt_path)
    print('=> Removed optimizer and lr scheduler of ckpt {} and save it to {}'.format(ori_ckpt_path, new_ckpt_path))


def get_optim_lr_str(optimizer):
    lr_strs = ['{:.6f}'.format(g['lr']).rstrip('0') for g in optimizer.param_groups]
    lr_str = ', '.join(lr_strs)
    return lr_str

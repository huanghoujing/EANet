"""This is the config for GlobalPool.
Some config may be reconfigured and new items may be added at run time."""

from __future__ import print_function
from easydict import EasyDict

cfg = EasyDict()

cfg.model = EasyDict()

cfg.model.backbone = EasyDict()
cfg.model.backbone.name = 'resnet50'
cfg.model.backbone.last_conv_stride = 1
cfg.model.backbone.pretrained = True
cfg.model.backbone.pretrained_model_dir = 'imagenet_model'

cfg.model.pool_type = 'GlobalPool'  # ['GlobalPool', 'PCBPool', 'PAPool']
cfg.model.max_or_avg = 'max'
cfg.model.em_dim = 512  #
cfg.model.num_parts = 1  #
cfg.model.use_ps = False  #

cfg.model.ps_head = EasyDict()
cfg.model.ps_head.mid_c = 256
cfg.model.ps_head.num_classes = 8

cfg.dataset = EasyDict()
cfg.dataset.root = 'dataset'

cfg.dataset.im = EasyDict()
cfg.dataset.im.h_w = (256, 128)  # final size for network input
# https://pytorch.org/docs/master/torchvision/models.html#torchvision-models
cfg.dataset.im.mean = [0.486, 0.459, 0.408]
cfg.dataset.im.std = [0.229, 0.224, 0.225]

cfg.dataset.use_pap_mask = False  #
cfg.dataset.pap_mask = EasyDict()
cfg.dataset.pap_mask.h_w = (24, 8)  # final size for masking
cfg.dataset.pap_mask.type = 'PAP_9P'

cfg.dataset.use_ps_label = False  #
cfg.dataset.ps_label = EasyDict()
cfg.dataset.ps_label.h_w = (48, 16)  # final size for calculating loss

# Note that cfg.dataset.train.* will not be accessed directly. Intended behavior e.g.
#     from package.utils.cfg import transfer_items
#     transfer_items(cfg.dataset.train, cfg.dataset)
#     print(cfg.dataset.transform_list)
# Similar for cfg.dataset.test.*, cfg.dataloader.train.*, cfg.dataloader.test.*
cfg.dataset.train = EasyDict()
cfg.dataset.train.name = 'market1501'  # ['market1501', 'cuhk03_np_detected_jpg', 'duke']
cfg.dataset.train.split = 'train'  #
cfg.dataset.train.transform_list = ['hflip', 'resize']

cfg.dataset.cd_train = EasyDict()
cfg.dataset.cd_train.name = 'duke'  #
cfg.dataset.cd_train.split = 'train'  #
cfg.dataset.cd_train.transform_list = ['hflip', 'resize']

cfg.dataset.test = EasyDict()
cfg.dataset.test.names = ['market1501', 'cuhk03_np_detected_jpg', 'duke']
if hasattr(cfg.dataset.test, 'query_splits'):
    assert len(cfg.dataset.test.query_splits) == len(cfg.dataset.test.names), "If cfg.dataset.test.query_splits is defined, it should be set for each test set."
cfg.dataset.test.transform_list = ['resize']

cfg.dataloader = EasyDict()
cfg.dataloader.num_workers = 2

cfg.dataloader.train = EasyDict()
cfg.dataloader.train.batch_type = 'random'
cfg.dataloader.train.batch_size = 32
cfg.dataloader.train.drop_last = True

cfg.dataloader.cd_train = EasyDict()
cfg.dataloader.cd_train.batch_type = 'random'
cfg.dataloader.cd_train.batch_size = 32
cfg.dataloader.cd_train.drop_last = True

cfg.dataloader.test = EasyDict()
cfg.dataloader.test.batch_type = 'seq'
cfg.dataloader.test.batch_size = 32
cfg.dataloader.test.drop_last = False

cfg.dataloader.pk = EasyDict()
cfg.dataloader.pk.k = 4

cfg.eval = EasyDict()
cfg.eval.forward_type = 'reid'
cfg.eval.chunk_size = 1000
cfg.eval.separate_camera_set = False
cfg.eval.single_gallery_shot = False
cfg.eval.first_match_break = True
cfg.eval.score_prefix = ''

cfg.train = EasyDict()

cfg.id_loss = EasyDict()
cfg.id_loss.name = 'idL'
cfg.id_loss.weight = 1  #
cfg.id_loss.use = cfg.id_loss.weight > 0

cfg.tri_loss = EasyDict()
cfg.tri_loss.name = 'triL'
cfg.tri_loss.weight = 0  #
cfg.tri_loss.use = cfg.tri_loss.weight > 0
cfg.tri_loss.margin = 0.3
cfg.tri_loss.dist_type = 'euclidean'
cfg.tri_loss.hard_type = 'tri_hard'
cfg.tri_loss.norm_by_num_of_effective_triplets = False

# source domain ps loss
cfg.src_ps_loss = EasyDict()
cfg.src_ps_loss.name = 'psL'
cfg.src_ps_loss.weight = 0  #
cfg.src_ps_loss.use = cfg.src_ps_loss.weight > 0
cfg.src_ps_loss.normalize_size = True
cfg.src_ps_loss.num_classes = cfg.model.ps_head.num_classes

# cross-domain (COCO/target domain) ps loss
cfg.cd_ps_loss = EasyDict()
cfg.cd_ps_loss.name = 'cd_psL'
cfg.cd_ps_loss.weight = 0  #
cfg.cd_ps_loss.use = cfg.cd_ps_loss.weight > 0
cfg.cd_ps_loss.normalize_size = True
cfg.cd_ps_loss.num_classes = cfg.model.ps_head.num_classes

cfg.log = EasyDict()
cfg.log.use_tensorboard = True

cfg.optim = EasyDict()
cfg.optim.optimizer = 'sgd'

cfg.optim.sgd = EasyDict()
cfg.optim.sgd.momentum = 0.9
cfg.optim.sgd.nesterov = False

cfg.optim.weight_decay = 5e-4
cfg.optim.ft_lr = 0.01  #
cfg.optim.new_params_lr = 0.02  #
cfg.optim.lr_decay_epochs = (25, 50)  #
cfg.optim.normal_epochs = 60  # Not including warmup/pretrain
cfg.optim.warmup_epochs = 0
cfg.optim.warmup = cfg.optim.warmup_epochs > 0
cfg.optim.warmup_init_lr = 0
cfg.optim.pretrain_new_params_epochs = 0  #
cfg.optim.pretrain_new_params = cfg.optim.pretrain_new_params_epochs > 0
cfg.optim.epochs_per_val = 5
cfg.optim.steps_per_log = 50
cfg.optim.trial_run = False  #
cfg.optim.phase = 'normal'  # [pretrain, warmup, normal], may be re-configured in code
cfg.optim.resume = False
cfg.optim.cft = False  #
cfg.optim.cft_iters = 1  #
cfg.optim.cft_rho = 8e-4
cfg.only_test = False  #
cfg.only_infer = False  #
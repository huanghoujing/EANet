"""Example of eanet_trainer.infer_dataloader(loader).
python script/exp/infer_dataloader_example.py
"""
from __future__ import print_function


import sys
sys.path.insert(0, '.')
from copy import deepcopy
from package.optim.eanet_trainer import EANetTrainer
from package.eval.eval_dataloader import eval_feat


def _print_stat(dic):
    print('=> Eval Statistics:')
    print('\tdic.keys():', dic.keys())
    print("\tdic['q_feat'].shape:", dic['q_feat'].shape)
    print("\tdic['q_label'].shape:", dic['q_label'].shape)
    print("\tdic['q_cam'].shape:", dic['q_cam'].shape)
    print("\tdic['g_feat'].shape:", dic['g_feat'].shape)
    print("\tdic['g_label'].shape:", dic['g_label'].shape)
    print("\tdic['g_cam'].shape:", dic['g_cam'].shape)


def main():
    from easydict import EasyDict

    args = EasyDict()
    args.exp_dir = 'exp/try_pcb_trained_on_market1501_for_reid_feature'  # There should be the corresponding `ckpt.pth` in it
    args.cfg_file = 'package/config/default.py'  # Set this `${EANet_PROJECT_DIR}`
    args.ow_file = 'paper_configs/PCB.txt'  # Set this `${EANet_PROJECT_DIR}`
    args.ow_str = "cfg.only_infer = True"

    eanet_trainer = EANetTrainer(args=args)
    q_loader = eanet_trainer.create_dataloader('test', name='market1501', split='query')
    g_loader = eanet_trainer.create_dataloader('test', name='market1501', split='gallery')
    q_feat = eanet_trainer.infer_dataloader(q_loader)
    g_feat = eanet_trainer.infer_dataloader(g_loader)
    dic = {
        'q_feat': q_feat['feat'],
        'q_label': q_feat['label'],
        'q_cam': q_feat['cam'],
        'g_feat': g_feat['feat'],
        'g_label': g_feat['label'],
        'g_cam': g_feat['cam'],
    }
    if 'visible' in q_feat:
        dic['q_visible'] = q_feat['visible']
    if 'visible' in g_feat:
        dic['g_visible'] = g_feat['visible']
    _print_stat(dic)
    return eval_feat(dic, deepcopy(eanet_trainer.cfg.eval))


if __name__ == '__main__':
    main()

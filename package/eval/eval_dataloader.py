from __future__ import print_function
from .extract_feat import extract_dataloader_feat
from .eval_feat import eval_feat


def _print_stat(dic):
    print('=> Eval Statistics:')
    print('\tdic.keys():', dic.keys())
    print("\tdic['q_feat'].shape:", dic['q_feat'].shape)
    print("\tdic['q_label'].shape:", dic['q_label'].shape)
    print("\tdic['q_cam'].shape:", dic['q_cam'].shape)
    print("\tdic['g_feat'].shape:", dic['g_feat'].shape)
    print("\tdic['g_label'].shape:", dic['g_label'].shape)
    print("\tdic['g_cam'].shape:", dic['g_cam'].shape)


def eval_dataloader(model, q_loader, g_loader, cfg):
    q_feat_dict = extract_dataloader_feat(model, q_loader, cfg)
    g_feat_dict = extract_dataloader_feat(model, g_loader, cfg)
    dic = {
        'q_feat': q_feat_dict['feat'],
        'q_label': q_feat_dict['label'],
        'q_cam': q_feat_dict['cam'],
        'g_feat': g_feat_dict['feat'],
        'g_label': g_feat_dict['label'],
        'g_cam': g_feat_dict['cam'],
    }
    if 'visible' in q_feat_dict:
        dic['q_visible'] = q_feat_dict['visible']
    if 'visible' in g_feat_dict:
        dic['g_visible'] = g_feat_dict['visible']
    _print_stat(dic)
    return eval_feat(dic, cfg)

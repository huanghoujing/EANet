from __future__ import print_function
import numpy as np
from sklearn.cluster import DBSCAN
from .eanet_trainer import EANetTrainer
from ..eval.extract_feat import extract_dataloader_feat
from ..eval.np_distance import compute_dist, compute_dist_with_visibility


class CFTTrainer(EANetTrainer):
    """Clustring and Finetuning."""

    def extract_feat(self):
        cfg = self.cfg
        extract_feat_loader = self.create_dataloader(mode='test', name=cfg.dataset.train.name, split=cfg.dataset.train.split)
        feat_dict = extract_dataloader_feat(self.model, extract_feat_loader, cfg.eval)
        return feat_dict

    def compute_dist(self, dic):
        if 'visible' in dic:
            dist_mat = compute_dist_with_visibility(dic['feat'], dic['feat'], dic['visible'], dic['visible'])
        else:
            dist_mat = compute_dist(dic['feat'], dic['feat'])
        return dist_mat

    def estimate_label(self, dist_mat):
        cfg = self.cfg
        print('dist_mat.min(), dist_mat.max(), dist_mat.mean(): ', dist_mat[dist_mat > 1e-6].min(), dist_mat[dist_mat > 1e-6].max(), dist_mat[dist_mat > 1e-6].mean())
        tri_mat = np.triu(dist_mat, 1)  # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
        tri_mat = np.sort(tri_mat, axis=None)
        top_num = np.round(cfg.optim.cft_rho * tri_mat.size).astype(int)
        # NOTE: DBSCAN eps requires positive value
        eps = tri_mat[:top_num].mean()
        print('eps in cluster: {:.3f}'.format(eps))
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)
        labels = cluster.fit_predict(dist_mat)
        return labels

    def select_pseudo_samples(self, feat_dict, labels):
        samples = []
        for idx, l in enumerate(labels):
            if l != -1:
                sample = {k: v[idx] for k, v in feat_dict.items()}
                sample['label'] = l
                del sample['cam']
                samples.append(sample)
        return samples

    def cft_one_iter(self):
        cfg = self.cfg.optim
        feat_dict = self.extract_feat()
        dist_mat = self.compute_dist(feat_dict)
        labels = self.estimate_label(dist_mat)
        num_labels = len(set(labels)) - 1
        assert num_labels > 1, "No Pseudo Labels Available!"
        print('NO. Pseudo Labels: {}'.format(num_labels))
        samples = self.select_pseudo_samples(feat_dict, labels)
        for phase in ['pretrain', 'normal']:
            cfg.phase = phase
            cfg.dont_test = phase == 'pretrain'
            self.init_trainer(samples=samples)
            self.load_items(model=True)
            self.train()

    def cft_total_iters(self):
        cfg = self.cfg
        # Test direct transfer of source domain -> target domain.
        # NOTE: ckpt.pth should be placed under current exp dir.
        self.load_items(model=True)
        self.test()
        for i in range(cfg.optim.cft_iters):
            print('*' * 50 + '\n' + 'Clustering and Finetuning Iter {}'.format(i + 1) + '\n' + '*' * 50)
            self.cft_one_iter()


if __name__ == '__main__':
    from ..utils import init_path
    trainer = CFTTrainer()
    if trainer.cfg.only_test:
        trainer.test()
    else:
        trainer.cft_total_iters()

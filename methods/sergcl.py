# DL and GL
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor

# Own modules
from methods.replay import Replay
from backbones.gcn import GCN
from backbones.encoder import Encoder
from methods.utility import get_graph_class_ratio
from backbones.gnn import train_node_classifier

# Utilities
from torch_geometric.loader import NeighborLoader
from .utility import *
import random
# from progressbar import progressbar


class SERGCL(Replay):
    def __init__(self, model, tasks, budget, m_update, device, args):
        super().__init__(model, tasks, budget, m_update, device)
        self.n_encoders = args['n_encoders']
        self.hid_dim = args['hid_dim']
        self.emb_dim = args['emb_dim']
        self.n_layers = args['n_layers']
        self.feat_init = "randomChoice"
        self.feat_init = args["feat_init"]
        self.hop = args['hop']
        self.activation = args['activation']
        self.n_samples = args['n_samples']
        self.mu_lr = args['mu_lr']
        self.std_lr = args['std_lr']

    def memorize(self, task, budgets):
        labels_cond = []
        for i, cls in enumerate(task.classes):
            labels_cond += [cls] * budgets[i]
        labels_cond = torch.tensor(labels_cond)
        mu_cond = torch.nn.Parameter(torch.FloatTensor(sum(budgets), task.num_features).to(self.device))
        mu_cond = self._initialize_feature(task, budgets, mu_cond, self.feat_init)
        std_cond = torch.nn.Parameter(0.001 * torch.ones(sum(budgets), task.num_features).to(self.device))

        mu_cond, std_cond, labels_cond = self._condense(task, mu_cond, std_cond, labels_cond, budgets)
        return [mu_cond, std_cond, labels_cond]
    
    def _initialize_feature(self, task, budgets, feat_cond, method="randomChoice"):
        if method == "randomNoise":
            torch.nn.init.xavier_uniform_(feat_cond)
        elif method == "randomChoice":
            sampled_ids = []
            for i, cls in enumerate(task.classes):
                train_mask = task.train_mask
                train_mask_at_cls = (task.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                sampled_ids += random.choices(ids_at_cls, k=budgets[i])
            sampled_feat = task.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat)
        elif method == "kMeans":
            sampled_ids = []
            for i, cls in enumerate(task.classes):
                train_mask = task.train_mask
                train_mask_at_cls = (task.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                sampled_ids += query(task, ids_at_cls, budgets[i], self.device)
            sampled_feat = task.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat)
        elif method == 'CM':
            budget_dist_compute = 1000
            d = 0.5
            feat_data = []
            for i, cls in enumerate(task.classes):
                train_mask = task.train_mask
                train_mask_at_cls = (task.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                vecs_at_cls = task.x[ids_at_cls]
                
                other_cls = (cls + 1) % len(task.classes)
                train_mask_at_other_cls = (task.y == other_cls).logical_and(train_mask)
                ids_at_other_cls = train_mask_at_other_cls.nonzero(as_tuple=True)[0].tolist()
                vecs_at_other_cls = task.x[ids_at_other_cls]

                random_vecs_at_other_cls = task.x[
                    random.choices(
                        ids_at_other_cls, k=min(budget_dist_compute, len(ids_at_other_cls))
                    )
                ]

                dist = torch.cdist(vecs_at_cls, random_vecs_at_other_cls.float()).half()
                
                n_selected = (dist < d).sum(dim=-1)
                rank = n_selected.sort()[1].tolist()
                current_ids_selected = rank[:budgets[i]]
                feat_data.append(vecs_at_cls[current_ids_selected])
            feat_data = torch.cat(feat_data, dim=0)
            feat_cond.data.copy_(feat_data)

        return feat_cond
    
    def _reparameterize(self, mu, std):
        std = std.relu() + 1e-8
        eps = torch.randn_like(mu)
        return eps.mul(std).add_(mu)

    def _condense(self, task, mu_cond, std_cond, labels_cond, budgets):
        opt_mu = torch.optim.Adam([mu_cond], lr=self.mu_lr)
        # opt_std = torch.optim.Adam([std_cond], lr=0.003)
        opt_std = torch.optim.Adam([std_cond], lr=self.std_lr)

        cls_train_masks = []
        for cls in task.classes:
            cls_train_masks.append((task.y == cls).logical_and(task.train_mask))   
        
        encoder = Encoder(task.num_features, self.hid_dim, self.emb_dim, self.n_layers, self.hop, self.activation).to(self.device)

        for _ in range(self.n_encoders):
            feat_cond = []
            lab_cond = []
            counter = 0
            for i, cls in enumerate(task.classes):
                cond_cls_mask = labels_cond == cls
                lab_cond += [labels_cond[cond_cls_mask].repeat(self.n_samples,)]
                cls_specific_feat_cond = self._reparameterize(
                    mu_cond[cond_cls_mask].repeat(self.n_samples, 1), 
                    std_cond[cond_cls_mask].repeat(self.n_samples, 1)
                )
                feat_cond += [cls_specific_feat_cond]
                counter += budgets[i]
            feat_cond = torch.cat(feat_cond, dim=0)
            lab_cond = torch.cat(lab_cond, dim=0)
            N_ENC = 1
            for enc in range(N_ENC):
                encoder.initialize()
                with torch.no_grad():
                    if enc == 0:
                        pred_real = encoder.encode(task.x.to(self.device), task.adj_t.to(self.device))
                        emb_real = [F.normalize(h) / N_ENC for h in encoder.h]
                    else:
                        pred_real = encoder.encode(task.x.to(self.device), task.adj_t.to(self.device))
                        for i, h in enumerate(encoder.h):
                            emb_real[i] = emb_real[i] + F.normalize(h) / N_ENC
                if enc == 0:
                    pred_cond = encoder.encode_without_e(feat_cond.to(self.device))
                    emb_cond = [F.normalize(h) / N_ENC for h in encoder.h]
                else:
                    pred_cond = encoder.encode_without_e(feat_cond.to(self.device))
                    for i, h in enumerate(encoder.h):
                        emb_cond[i] = emb_cond[i] + F.normalize(h) / N_ENC
            
            loss = torch.tensor(0.).to(self.device)
            for e_r, e_c in zip(emb_real, emb_cond):
                for i, cls in enumerate(task.classes):
                    real_emb_at_class = e_r[cls_train_masks[i]]
                    cond_emb_at_class = e_c[lab_cond == cls]
                    
                    mean_dist = torch.mean(real_emb_at_class, 0).detach() - \
                        torch.mean(cond_emb_at_class, 0)
                    loss += torch.sum(mean_dist ** 2)

            # Update the feature matrix            
            opt_mu.zero_grad()
            opt_std.zero_grad()
            loss.backward()
            opt_mu.step()
            opt_std.step()

        return mu_cond.detach(), std_cond.detach(), labels_cond.detach().to(int)
    

'''
CUDA_VISIBLE_DEVICES=3 python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../CaT_data --result-path ./results --dataset-name arxiv --budget 29 --cgm-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}"
'''
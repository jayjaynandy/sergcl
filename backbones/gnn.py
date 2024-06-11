import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.data import Batch
# from focal_loss.focal_loss import FocalLoss


adj_method = 'self_loop'


def generate_nodes(mu, std):
    std = std.relu() + 1e-8
    eps = torch.randn_like(mu)
    return eps.mul(std).add_(mu)


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

    def initialize(self):
        for layer in self.layers:
            # torch.nn.init.normal_(layer.lin.weight.data)
            layer.reset_parameters()
    
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
        x = self.layers[-1](x, adj_t)
        return x
        # return F.log_softmax(x, dim=1)

    def encode(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            x = layer(x, adj_t)
            x = F.relu(x)
        return x
    
    def encode_noise(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_t)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        random_noise = torch.rand_like(x).cuda()
        x += torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        return x

def train_node_classifier(model, memory_bank, tasks, optimizer, k, weight=None, n_epoch=200, args=None):
    # import wandb
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="CaT"
    # )
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    t = k+1 if args.tim else k
    for epoch in range(n_epoch):
        previous_graphs = memory_bank[:t]
        replayed_graphs = []
        
        for i in range(t):
            feat_cond = []
            labels_cond = []

            previous_mu = previous_graphs[i][0]
            previous_std = previous_graphs[i][1]
            previous_labels = previous_graphs[i][2]

            cls = torch.unique(previous_labels)
            
            for c in cls:
                c_mask = previous_labels == c
                if args.dataset_name == 'corafull':
                    n_repeat = 5
                    r_samples = 1000
                else:
                    n_repeat = 500
                    # r_samples = 250
                th = 0.995
                    
                labels_cond += [c] * n_repeat * torch.sum(c_mask)
                cls_specific_feat_cond = generate_nodes(
                    previous_mu[c_mask].repeat(n_repeat, 1).to(args.device), 
                    previous_std[c_mask].repeat(n_repeat, 1).to(args.device)
                )
                feat_cond += [cls_specific_feat_cond]
            feat_cond = torch.cat(feat_cond, dim=0)
            labels_cond = torch.tensor(labels_cond)

            # r_idx = torch.randint(0, feat_cond.shape[0], (r_samples,))
            # feat_cond = feat_cond[r_idx]
            # labels_cond = labels_cond[r_idx]

            if adj_method == 'th':
                feat_cond_norm = torch.nn.functional.normalize(feat_cond, dim=1)
                adj_t = torch.matmul(feat_cond_norm, feat_cond_norm.t())
                adj_t[adj_t < th] = 0
                adj_t[adj_t >= th] = 1
                index = torch.nonzero(adj_t).t()
                value = torch.ones(index.shape[1], dtype=float, device=args.device)
                adj_t = SparseTensor(
                    row=index[0],                                                                  
                    col=index[1],
                    value=value
                )
            elif adj_method == 'bernoulli':
                if args.dataset_name == 'corafull':
                    factor = 0.001
                else:
                    factor = 0.00001
                feat_cond_norm = torch.nn.functional.normalize(feat_cond, dim=1)
                adj_t = torch.matmul(feat_cond_norm, feat_cond_norm.t()) / factor
                adj_t = factor * ((adj_t - adj_t.min())/ adj_t.max())
                adj_t = adj_t - torch.diag(torch.diag(adj_t, 0)) + \
                    torch.eye(adj_t.shape[0]).to(args.device)
                rand_adj_t = torch.bernoulli(adj_t)
                index = torch.nonzero(rand_adj_t).t()
                value = torch.ones(index.shape[1], dtype=float, device=args.device)
                adj_t = SparseTensor(
                    row=index[0],
                    col=index[1],
                    value=value
                )
            elif adj_method == 'self_loop':
                adj_t = SparseTensor.eye(feat_cond.shape[0], feat_cond.shape[0]).t()

            previous_graph = Data(
                x=feat_cond.detach().cpu(), y=labels_cond, adj_t=adj_t)
            previous_graph.train_mask = torch.ones(feat_cond.shape[0], dtype=torch.bool)

            replayed_graphs.extend([previous_graph.to('cpu')])
        if not args.tim:
            replayed_graphs.extend([tasks[k]])
        replayed_graphs = Batch.from_data_list(replayed_graphs)
        replayed_graphs.to(args.device, "x", "y", "adj_t")
        max_cls = torch.unique(replayed_graphs.y)[-1].to(int)
        incremental_cls=(0, max_cls+1)
        if incremental_cls:
            out = model(replayed_graphs)[:, 0:incremental_cls[1]]
        else:
            out = model(replayed_graphs) 
        
        loss = ce(out[replayed_graphs.train_mask], replayed_graphs.y[replayed_graphs.train_mask].to(int))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss_train = ce(out[data.train_mask], data.y[data.train_mask])
        # loss_val = ce(out[data.val_mask], data.y[data.val_mask])
        # loss_test = ce(out[data.test_mask], data.y[data.test_mask])
        # wandb.log({"loss_train": loss_train, 
        #            "loss_val": loss_val, 
        #            "loss_test": loss_test})
    return model, max_cls

def train_node_classifier_batch(args, model, batches, optimizer, n_epoch=200, incremental_cls=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(n_epoch):
        for data in batches:
            data = data.to(args.device, "x", "y", "adj_t")
            if incremental_cls:
                out = model(data)[:, 0:incremental_cls[1]]
            else:
                out = model(data)

            loss = ce(out[data.train_mask], data.y[data.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def eval_node_classifier(model, data, incremental_cls=None):
    model.eval()
    pred = model(data)[data.test_mask, incremental_cls[0]:incremental_cls[1]].argmax(dim=1)
    correct = (pred == data.y[data.test_mask]-incremental_cls[0]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

# For graph learning
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_sparse import SparseTensor

# Utility
import os
import argparse
import sys
import numpy as np
from utilities import *
from data_stream import Streaming
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, train_node_classifier_batch, eval_node_classifier


# adj_method = 'bernoulli'
# adj_method = 'th'


def generate_nodes(mu, std):
    std = std.relu() + 1e-8
    eps = torch.randn_like(mu)
    return eps.mul(std).add_(mu)


def evaluate(args, dataset, data_stream, memory_banks, flush=True):
    APs = []
    AFs = []
    mAPs = []
    Ps = []
    for i in range(args.repeat):
        memory_bank = memory_banks[i]
        # Initialize the performance matrix.
        performace_matrix = torch.zeros(len(memory_bank), len(memory_bank))
        model = get_backbone_model(dataset, data_stream, args)
        cgl_model = get_cgl_model(model, data_stream, args)
        tasks = cgl_model.tasks

        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        mAP = 0
        for k in range(len(memory_bank)):
            # train
            if args.dataset_name == "products" and args.cgl_method == "joint":
                max_cls = torch.unique(memory_bank[k].y)[-1]
                batches = memory_bank[:k+1]
                for data in batches:
                    data.to(args.device)
                model = train_node_classifier_batch(model, batches, opt, n_epoch=args.cls_epoch, incremental_cls=(0, max_cls+1))
            else:
                # if args.tim:
                #     if args.batch:
                #         replayed_graphs = memory_bank[:k+1]
                #     else:
                #         replayed_graphs = Batch.from_data_list(memory_bank[:k+1])
                # else:
                #     if args.batch:
                #         replayed_graphs = memory_bank[:k] + [tasks[k]]
                #     else:
                #         replayed_graphs = Batch.from_data_list(memory_bank[:k] + [tasks[k]])

                # previous_graphs = memory_bank[:k]
                # current_graph = [tasks[k]]
                # replayed_graphs = []
                # current_samples = tasks[k].x.shape[0]
                
                # for i in range(k+1):
                #     feat_cond = []
                #     labels_cond = []

                #     previous_mu = previous_graphs[i][0]
                #     previous_std = previous_graphs[i][1]
                #     previous_labels = previous_graphs[i][2]

                #     cls = torch.unique(previous_labels)
                    
                #     for c in cls:
                #         c_mask = previous_labels == c
                #         if args.dataset_name.lower() == 'arxiv':
                #             n_repeat = 5
                #             th = 0.995
                            
                #         elif args.dataset_name.lower() == 'corafull':
                #             n_repeat = 5
                #             th = 0.995
                #             # n_samples = min(10500, 20 * current_samples) // torch.sum(c_mask)
                #             # th = 0.995
                #             # factor = 0.15
                #         elif args.dataset_name.lower() == 'products':
                #             n_repeat = 5
                #             th = 0.995
                            
                #         labels_cond += [c] * n_repeat * torch.sum(c_mask)
                #         cls_specific_feat_cond = generate_nodes(
                #             previous_mu[c_mask].repeat(n_repeat, 1).to(args.device), 
                #             previous_std[c_mask].repeat(n_repeat, 1).to(args.device)
                #         )
                #         feat_cond += [cls_specific_feat_cond]
                #     feat_cond = torch.cat(feat_cond, dim=0)
                #     labels_cond = torch.tensor(labels_cond)

                #     # self_loops = SparseTensor.eye(feat_cond.shape[0], feat_cond.shape[0]).t()
                #     if adj_method == 'th':
                #         feat_cond_norm = torch.nn.functional.normalize(feat_cond, dim=1)
                #         adj_t = torch.matmul(feat_cond_norm, feat_cond_norm.t())
                #         # adj_t = (adj_t - adj_t.min()) / (adj_t.max() - adj_t.min())
                #         adj_t[adj_t < th] = 0
                #         adj_t[adj_t >= th] = 1
                #         index = torch.nonzero(adj_t).t()
                #         value = torch.ones(index.shape[1], dtype=float, device=args.device)
                #         self_loops = SparseTensor(
                #             row=index[0],                                                                  
                #             col=index[1],
                #             value=value
                #         )
                #     elif adj_method == 'bernoulli':
                #         feat_cond_norm = torch.nn.functional.normalize(feat_cond, dim=1)
                #         adj_t = torch.matmul(feat_cond_norm, feat_cond_norm.t())
                #         adj_t = (adj_t - adj_t.min()) / (adj_t.max() - adj_t.min())
                #         # adj_t = adj_t * factor - torch.diag(torch.diag(adj_t * factor, 0)) + torch.eye(adj_t.shape[0]).to(args.device)
                #         adj_t = torch.nn.ReLU()(adj_t - factor) - torch.diag(torch.diag(adj_t - factor, 0)) + \
                #             torch.eye(adj_t.shape[0]).to(args.device)
                        
                #         rand_adj_t = torch.bernoulli(adj_t)
                #         index = torch.nonzero(rand_adj_t).t()
                #         value = torch.ones(index.shape[1], dtype=float, device=args.device)
                #         self_loops = SparseTensor(
                #             row=index[0],
                #             col=index[1],
                #             value=value
                #         )

                #     # print(self_loops)
                    
                #     previous_graph = Data(
                #         x=feat_cond.detach().cpu(), y=labels_cond, adj_t=self_loops)
                #     previous_graph.train_mask = torch.ones(feat_cond.shape[0], dtype=torch.bool)

                #     replayed_graphs.extend([previous_graph.to('cpu')])
                # # replayed_graphs.extend(current_graph)
                # replayed_graphs = Batch.from_data_list(replayed_graphs)

                if args.batch:
                    max_cls = torch.unique(memory_bank[k].y)[-1]
                    batches = replayed_graphs
                    # for data in batches:
                    #     data.to(args.device)
                    model = train_node_classifier_batch(
                        args, model, batches, opt, n_epoch=args.cls_epoch, 
                        incremental_cls=(0, max_cls+1)
                    )
                else:
                    # replayed_graphs.to(args.device, "x", "y", "adj_t")
                    # max_cls = torch.unique(replayed_graphs.y)[-1].to(int)
                    model, max_cls = train_node_classifier(model, memory_bank, tasks, opt, k, weight=None, n_epoch=args.cls_epoch, args=args)
            torch.cuda.empty_cache()
            # Test the model from task 0 to task k
            accs = []
            AF = 0
            for k_ in range(k + 1):
                task_ = tasks[k_].to(args.device, "x", "y", "adj_t")
                if args.IL == "classIL":
                    acc = eval_node_classifier(model, task_, incremental_cls=(0, max_cls+1)) * 100
                else:
                    max_cls = torch.unique(task_.y)[-1]
                    acc = eval_node_classifier(model, task_, incremental_cls=(max_cls+1-data_stream.cls_per_task, max_cls+1)) * 100
                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}", end="|", flush=flush)
                performace_matrix[k, k_] = acc
            AP = sum(accs) / len(accs)
            mAP += AP
            print(f"AP: {AP:.2f}", end=", ", flush=flush)
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=flush)
        APs.append(AP)
        AFs.append(AF)
        mAPs.append(mAP/(k+1))
        Ps.append(performace_matrix)
    print(f"AP: {np.mean(APs):.1f}±{np.std(APs, ddof=1):.1f}", flush=flush)
    print(f"mAP: {np.mean(mAPs):.1f}±{np.std(mAPs, ddof=1):.1f}", flush=flush)
    print(f"AF: {np.mean(AFs):.1f}±{np.std(AFs, ddof=1):.1f}", flush=flush)
    return Ps

def main():
    parser = argparse.ArgumentParser()
    # Arguments for data.
    parser.add_argument('--dataset-name', type=str, default="corafull")
    parser.add_argument('--cls-per-task', type=int, default=2)
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--result-path', type=str, default="./results")

    # Argumnets for CGL methods.
    parser.add_argument('--tim', action='store_true')
    parser.add_argument('--cgl-method', type=str, default="cgm")
    parser.add_argument('--cls-epoch', type=int, default=200)
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--m-update', type=str, default="all")
    parser.add_argument('--sergcl-args', type=str, default="{}")
    parser.add_argument('--cgm-args', type=str, default="{}")
    parser.add_argument('--ewc-args', type=str, default="{'memory_strength': 100000.}")
    parser.add_argument('--mas-args', type=str, default="{'memory_strength': 10000.}")
    parser.add_argument('--gem-args', type=str, default="{'memory_strength': 0.5, 'n_memories': 20}")
    parser.add_argument('--twp-args', type=str, default="{'lambda_l': 10000., 'lambda_t': 10000., 'beta': 0.01}")
    parser.add_argument('--lwf-args', type=str, default="{'lambda_dist': 10., 'T': 20.}")
    parser.add_argument('--IL', type=str, default="classIL")
    parser.add_argument('--batch', action='store_true')

    # Others
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--rewrite', action='store_true')

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    # Get file names.
    result_file_name = get_result_file_name(args)
    os.makedirs(os.path.join(args.result_path, "memory_bank"), exist_ok=True)
    memory_bank_file_name = os.path.join(args.result_path, "memory_bank" , result_file_name)
    task_file = os.path.join(args.data_dir, "streaming", f"{args.dataset_name}.streaming")

    dataset = get_dataset(args)
    if os.path.exists(task_file):
        data_stream = torch.load(task_file)
    else:
        data_stream = Streaming(args.cls_per_task, dataset)
        torch.save(data_stream, task_file)

    if args.cgl_method in ["bare", "ewc", "mas", "gem", "twp", "lwf"]:
        APs = []
        AFs = []
        mAPs = []
        for i in range(args.repeat):
            model = get_backbone_model(dataset, data_stream, args)
            cgl_model = get_cgl_model(model, data_stream, args)
            AP, mAP, AF = cgl_model.observer(args.cls_epoch, args.IL)
            APs.append(AP)
            AFs.append(AF)
            mAPs.append(mAP)
        print(f"AP: {np.mean(APs):.1f}±{np.std(APs, ddof=1):.1f}", flush=True)
        print(f"mAP: {np.mean(mAPs):.1f}±{np.std(mAPs, ddof=1):.1f}", flush=True)
        print(f"AF: {np.mean(AFs):.1f}±{np.std(AFs, ddof=1):.1f}", flush=True)
    else:
        # Get memory banks.
        memory_banks = []
        for i in range(args.repeat):
            if os.path.exists(memory_bank_file_name + f"_repeat_{i}") and not args.rewrite:
                memory_bank = torch.load(memory_bank_file_name + f"_repeat_{i}")
                memory_banks.append(memory_bank)  # load the memory bank from the file.
            else:
                model = get_backbone_model(dataset, data_stream, args)
                cgl_model = get_cgl_model(model, data_stream, args)

                memory_bank = cgl_model.observer()
                memory_banks.append(memory_bank)
                torch.save(memory_bank, memory_bank_file_name + f"_repeat_{i}")
        
        Ps = evaluate(args, dataset, data_stream, memory_banks)
        os.makedirs(os.path.join(args.result_path, "performance_new"), exist_ok=True)
        if args.tim:
            if args.batch:
                torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_tim_batch.pt"))
            else:
                torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_tim.pt"))
        else:
            if args.batch:
                torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}_batch.pt"))
            else:
                torch.save(Ps, os.path.join(args.result_path, "performance_new", f"{result_file_name}.pt"))


if __name__ == '__main__':
    main()
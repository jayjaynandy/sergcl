import torch
import numpy as np
import matplotlib.pyplot as plt


ds = 'products'
budget = 318
ps = torch.load(f'/DATA/arnab/CaT-CGL-StoreStats/results/performance_new/{ds}_{budget}_cgm_normal_randomChoice_feat_0.001_1000_2_layer_512_GCN_hop_1_tim.pt')
ps = ps[0].numpy()
# Create a mask
mask = np.triu((np.ones_like(ps) - np.eye(ps.shape[0])).astype(bool))
plt_mat = np.ma.array(ps, mask=mask)

fig = plt.figure(figsize=(4.74, 4.82))
ax1 = fig.add_subplot(111)
cmap = plt.get_cmap()
cmap.set_bad('w')
ax1.matshow(plt_mat)
ax1.xaxis.set_ticks_position("bottom")
ax1.xaxis.set_label_position("bottom")
ax1.set_frame_on(False)
ax1.set_xlabel('Tasks', fontsize=18)
ax1.set_ylabel('Tasks', fontsize=18)
if ds == 'corafull':
    ax1.set_xticks([0, 10, 20, 30])
    ax1.set_xticklabels([0, 10, 20, 30])
    ax1.set_yticks([0, 10, 20, 30])
    ax1.set_yticklabels([0, 10, 20, 30])
if ds == 'arxiv' or ds == 'products':
    ax1.set_xticks([0, 10, 20])
    ax1.set_xticklabels([0, 10, 20])
    ax1.set_yticks([0, 10, 20])
    ax1.set_yticklabels([0, 10, 20])
params = {'axes.labelsize': 16, 'axes.titlesize': 16}
plt.rcParams.update(params)
plt.tight_layout()
plt.savefig(f'./{ds}.pdf')
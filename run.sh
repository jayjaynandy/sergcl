python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../data --result-path ./results --dataset-name arxiv --budget 29 --sergcl-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'n_samples': 100, 'mu_lr': 1e-3, 'std_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}"
python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../data --result-path ./results --dataset-name corafull --budget 4 --sergcl-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'n_samples': 100, 'mu_lr': 1e-3, 'std_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}"
python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../data --result-path ./results --dataset-name reddit --budget 40 --sergcl-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'n_samples': 100, 'mu_lr': 1e-3, 'std_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}"
python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../data --result-path ./results --dataset-name products --budget 318 --sergcl-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'n_samples': 100, 'mu_lr': 1e-3, 'std_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}"


python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../data --result-path ./results --dataset-name arxiv --budget 29 --sergcl-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'n_samples': 100, 'mu_lr': 1e-3, 'std_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}" --IL taskIL
python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../data --result-path ./results --dataset-name corafull --budget 4 --sergcl-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'n_samples': 100, 'mu_lr': 1e-3, 'std_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}" --IL taskIL
python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../data --result-path ./results --dataset-name reddit --budget 40 --sergcl-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'n_samples': 100, 'mu_lr': 1e-3, 'std_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}" --IL taskIL
python train.py --seed 1024 --repeat 1 --cls-epoch 500 --cgl-method sergcl --tim --data-dir ../data --result-path ./results --dataset-name products --budget 318 --sergcl-args "{'n_encoders': 1000, 'feat_init': 'randomChoice', 'n_samples': 100, 'mu_lr': 1e-3, 'std_lr': 1e-3, 'hid_dim': 512, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True}" --IL taskIL
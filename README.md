# GrMPN
Graph-based Motion Planning Networks

## Prerequisites

All dependencies can be installed by setting up a docker-container via`Dockerfile`. 

Unzip data from: https://drive.google.com/open?id=1FBrPhnDJxntCcZ2rg_Y2tedxMBJA4-Ld

## Running Experiments

Experiments for a particular environment can be run using:

GAT - Lattice 16x16 - samples 1000
```
python3 main_navigation.py --model GAT --name ./results/grid_gppn5k_1000_gat_16x16 --input_train ./data/lattice/gridworld_gppn_5k_16x16.npz --input_test ./data/lattice/gridworld_gppn_5k_16x16.npz --is_train=True --n_train 20 --n_ptrain 15 --n_ptest 15 --train_max_mem 1000 --valid_max_mem 200 --mem_size 200 --batch_train 20 --batch_test 200 --out_file ./results/gppn5k_1000_gat_16x16.csv
```

GAT - Train on Tree-like 50, valid on Tree-like 150
```
python3 main_navigation.py --model GAT --name ./results/ir_theta200_gat_50_150 --input_train ./data/irregular/irregular_theta200_50.pkl --input_test ./data/irregular/irregular_theta200_150.pkl --is_train=True --n_train 10 --n_ptrain 20 --n_ptest 20 --train_max_mem 0 --valid_max_mem 200 --mem_size 200 --batch_train 40 --batch_test 200 --out_file ./results/ir_theta200_gat_50_150.csv
```

GAT - Train on PRM 200, valid on PRM 200
```
python3 main_navigation.py --model GAT --name ./results/irregular_prm_8x8_200 --input_train ./data/motion_planning/irregular_prm_8x8_200.pkl --input_test ./data/motion_planning/irregular_prm_8x8_200.pkl --is_train=True --n_train 10 --n_ptrain 20 --n_ptest 20 --train_max_mem 0 --valid_max_mem 200 --mem_size 200 --batch_train 40 --batch_test 200 --out_file ./results/irregular_prm_8x8_200.csv
```

Results will be saved each iteration under the predefined `name` directory.

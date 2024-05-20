import itertools
import os
import shutil

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from gnn4bcprediction.ml_scheme import test_torch
from gnn4bcprediction.nn_models import GCN, GATv2, GraphSAGE
from gnn4bcprediction.nn_models import MLP

## 0. Set torch configurations ################################################

torch.set_default_tensor_type(torch.FloatTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## 1. Set paths and models/training configurations ############################
models_tuning_folder = 'models/tuning/'
models_best_folder = 'models/best/'
results_tuning_folder = 'data/tuning_results/'
results_best_folder = 'data/best_results/'
datasets_folder = 'data/datasets/synthetic/'
os.makedirs(models_best_folder, exist_ok=True)
os.makedirs(results_best_folder, exist_ok=True)

criterion = torch.nn.MSELoss()
layers = {'mlp': MLP, 'gcn': GCN, 'sage': GraphSAGE, 'gatv2': GATv2}
scenarios = ['hom', 'com']
lr_list = [0.01, 0.001, 0.0001]
L_list = [4, 5]
H_list = [16, 32]
bs_list = [2, 4, 8]

## 2. For each scenario/layer, test the corresponding models ##################
for scenario in scenarios:
    dataset_name = f'synthetic_1000000_10_20_0.5_{scenario}'
    # Datasets
    train_dataset = [data.to(device) for data in torch.load(
        f'{datasets_folder}{dataset_name}_0.2-0.2_train.pt')]
    val_dataset = [data.to(device) for data in torch.load(
        f'{datasets_folder}{dataset_name}_0.2-0.2_val.pt')]

    # Data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for layer_name in layers.keys():
        models_root = f'{models_tuning_folder}{dataset_name}/{layer_name}/'
        is_gnn = layer_name in ['gcn', 'sage', 'gatv2']

        results = pd.DataFrame(
            columns=['scenario', 'layer', 'lr', 'L', 'H', 'bs', 'mse_train',
                     'mse_val'])

        for lr, L, H, bs in itertools.product(lr_list, L_list, H_list,
                                              bs_list):
            config = f'{layer_name}_{scenario}_{lr}_{L}_{H}_b{bs}'

            model = layers[layer_name](2, H, 1, L - 2).to(device)
            model.load_state_dict(torch.load(f'{models_root}{config}.pt'))
            model.eval()

            mse_train = test_torch(model, train_data_loader, criterion, is_gnn)
            mse_val = test_torch(model, val_data_loader, criterion, is_gnn)

            results = pd.concat([results, pd.DataFrame(
                {'scenario': scenario, 'layer': layer_name, 'lr': lr, 'L': L,
                 'H': H, 'bs': bs, 'mse_train': mse_train, 'mse_val': mse_val},
                index=[0])], ignore_index=True)

        # 2a) Order results and save CSV ######################################
        results = results.sort_values(by='mse_val')
        results.to_csv(
            f'data/tuning_results/{dataset_name}/{layer_name}/{layer_name}_{scenario}_results.csv',
            index=False)

        # 2b) Move best model and its train loss IMG ##########################

        best_config = results.iloc[0]
        config = f'{layer_name}_{scenario}_{best_config["lr"]}_{best_config["L"]}_{best_config["H"]}_b{best_config["bs"]}'

        model_file = f'{models_root}{config}.pt'
        train_evo_img = f'data/tuning_results/{dataset_name}/{layer_name}/{config}.png'

        shutil.copy(model_file, f'{models_best_folder}')
        shutil.copy(train_evo_img,
                    f'{results_best_folder}/train_loss_{config}.png')

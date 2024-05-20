import os

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
    mean_absolute_percentage_error
from torch_geometric.loader import DataLoader

from gnn4bcprediction.nn_models import GCN, GATv2, GraphSAGE
from gnn4bcprediction.nn_models import MLP

## 0. Set torch configurations ################################################

torch.set_default_tensor_type(torch.FloatTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## 1. Set paths and models/training configurations ############################
models_best_folder = 'models/best/'
datasets_folder = 'data/datasets/'
test_results_root = 'data/test_results/'

layers = {'mlp': MLP, 'gcn': GCN, 'sage': GraphSAGE, 'gatv2': GATv2}
layer_title = {'mlp': 'MLP', 'gcn': 'GCN', 'sage': 'GraphSAGE',
               'gatv2': 'GATv2'}

sum_results = pd.DataFrame(
    columns=['scenario', 'dataset', 'layer', 'mse', 'mae', 'mape', 'r2'])

batch_results = {'hom': {}, 'com': {}}

for dataset_name in os.listdir(datasets_folder):
    test_results_folder = f'{test_results_root}{dataset_name}/'
    os.makedirs(test_results_folder, exist_ok=True)
    for dataset_scenario in os.listdir(f'{datasets_folder}{dataset_name}'):
        split = dataset_scenario.split('_')[-1].split('.')[0]
        if split not in ['train', 'val']:
            scenario = 'hom' if 'hom' in dataset_scenario else 'com'
            batch_results[scenario][dataset_name] = {}
            print(f'Processing dataset {dataset_name} - {scenario}')

            dataset = [data.to(device) for data in torch.load(
                f'{datasets_folder}{dataset_name}/{dataset_scenario}')]
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

            for model_file in os.listdir(models_best_folder):
                # Load the model ##############################################
                model_config = model_file.split('_')
                if model_config[1] == scenario:
                    layer_name = model_config[0]
                    lr = float(model_config[2])
                    L = int(model_config[3])
                    H = int(model_config[4])
                    bs = int(model_config[5].split('.')[0][1:])
                    is_gnn = layer_name in ['gcn', 'sage', 'gatv2']

                    model = layers[layer_name](2, H, 1, L - 2).to(device)
                    model.load_state_dict(
                        torch.load(f'{models_best_folder}{model_file}'))
                    model.eval()

                    print(
                        f'*** Processing model {model_file}: {layer_name} - {lr} - {L} - {H} - {bs}')

                    # Evaluate the model on every batch and node ##############
                    y_true = []
                    y_pred = []
                    for batch in data_loader:
                        edge_index = batch.edge_index if is_gnn else None
                        y_true.append(
                            batch.y.cpu().detach().numpy().reshape(-1))
                        y_pred.append(model.forward(batch.x,
                                                    edge_index).cpu().detach().numpy().reshape(
                            -1))

                    complete_y_true = np.concatenate(y_true)
                    complete_y_pred = np.concatenate(y_pred)

                    # 1a) Fill summary table with MSE, MAE, MAPE and R2 results
                    mse = mean_squared_error(complete_y_true, complete_y_pred)
                    mae = mean_absolute_error(complete_y_true, complete_y_pred)
                    mape = mean_absolute_percentage_error(complete_y_true,
                                                          complete_y_pred)
                    r2 = r2_score(complete_y_true, complete_y_pred)

                    sum_results = pd.concat([sum_results, pd.DataFrame(
                        {'scenario': scenario, 'dataset': dataset_name,
                         'layer': layer_name, 'mse': mse, 'mae': mae,
                         'mape': mape, 'r2': r2}, index=[0])],
                                            ignore_index=True)

                    # 2) Plot of predictions vs. real values ##################
                    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
                    plt.scatter(complete_y_true, complete_y_pred, alpha=0.1)

                    axs.set_xlabel('True threshold')

                    if layer_name == 'mlp':
                        axs.set_ylabel('Predicted threshold')
                    else:
                        axs.set_ylabel('Predicted threshold', color='white')

                    axs.set_ylim(-0.025, 0.525)
                    if scenario == 'hom':
                        axs.set_xlim(0.08, 0.525)
                        axs.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
                    else:
                        axs.set_xlim(-0.025, 0.525)
                        axs.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

                    axs.set_title(f'{layer_title[layer_name]}')

                    plt.savefig(f'{test_results_folder}{model_file[:-3]}.png')
                    plt.clf()

                    # 3a) Fill batch results for Statistical Tests ############
                    batch_results[scenario][dataset_name][layer_name] = [
                        mean_squared_error(batch_y_true, batch_y_pred) for
                        batch_y_true, batch_y_pred in zip(y_true, y_pred)]

# 1b) Save CSV with MSE, MAE, MAPE and R2 results #############################

scenario_order = ['hom', 'com']
dataset_order = ['synthetic', 'cora', 'cora_ml', 'citeseer', 'pubmed', 'dblp']
layer_order = ['mlp', 'gcn', 'sage', 'gatv2']

sum_results['scenario'] = pd.Categorical(sum_results['scenario'],
                                         categories=scenario_order)
sum_results['dataset'] = pd.Categorical(sum_results['dataset'],
                                        categories=dataset_order)
sum_results['layer'] = pd.Categorical(sum_results['layer'],
                                      categories=layer_order)
sum_results = sum_results.sort_values(by=['scenario', 'dataset', 'layer'])

sum_results.to_csv(f'{test_results_root}sum_results.csv', index=False)

# 3b) Perform Statistical tests for Synthetic and Real cases
batch_real_results = {'hom': {}, 'com': {}}
top_types = {'real': ['cora', 'cora_ml', 'citeseer', 'pubmed', 'dblp'],
             'syn': ['synthetic']}

for scenario in ['hom', 'com']:
    for top_type in top_types.keys():
        statistic_results = pd.DataFrame(
            columns=['test', 'pvalue', 'statistic'])

        os.makedirs(f'{test_results_root}stats_{top_type}/', exist_ok=True)
        global_stats_path = f'{test_results_root}stats_{top_type}/stats_{top_type}_{scenario}_global.csv'
        nemen_stats_path = f'{test_results_root}stats_{top_type}/stats_{top_type}_{scenario}_nemenyi.csv'

        graph_results = {}
        for layer_name in ['mlp', 'gcn', 'sage', 'gatv2']:
            graph_results[layer_name] = []
            for dataset_name in top_types[top_type]:
                graph_results[layer_name] += \
                    batch_results[scenario][dataset_name][layer_name]

            # Shapiro-Wilk test ###############################################
            test_shap = stats.shapiro(graph_results[layer_name])
            statistic_results = pd.concat([statistic_results, pd.DataFrame(
                {'test': f'shapiro-{layer_name}-{scenario}',
                 'pvalue': test_shap[1], 'statistic': test_shap[0]},
                index=[0])], ignore_index=True)

        # Friedman test #######################################################
        test_fried = stats.friedmanchisquare(
            *[graph_results[layer_name] for layer_name in
              ['mlp', 'gcn', 'sage', 'gatv2']])
        statistic_results = pd.concat([statistic_results, pd.DataFrame(
            {'test': f'friedman-{scenario}', 'pvalue': test_fried[1],
             'statistic': test_fried[0]}, index=[0])], ignore_index=True)
        statistic_results.to_csv(global_stats_path, index=False)

        # Post-hoc Nemenyi test ###############################################
        data = np.array([graph_results[layer_name] for layer_name in
                         ['mlp', 'gcn', 'sage', 'gatv2']])
        test_nemenyi = sp.posthoc_nemenyi_friedman(data.T)
        nemenyi_results = pd.DataFrame(test_nemenyi)
        nemenyi_results.columns = ['mlp', 'gcn', 'sage', 'gatv2']
        nemenyi_results.index = ['mlp', 'gcn', 'sage', 'gatv2']
        nemenyi_results.to_csv(nemen_stats_path, index=True)

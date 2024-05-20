import sys

import torch
from torch_geometric.loader import DataLoader

from gnn4bcprediction.ml_scheme import train_model, test_torch
from gnn4bcprediction.nn_models import MLP, GCN, GATv2, GraphSAGE

try:
    dataset_name = sys.argv[1]
    layer_name = sys.argv[2]
    lr = float(sys.argv[3])
    num_layers = int(sys.argv[4])
    hidden_dim = int(sys.argv[5])
    batch_size = int(sys.argv[6])
except IndexError:
    print("{0} <dataset_name> <layer_name> <lr> <num_layers> "
          "<hidden_dim> <batch_size>".format(sys.argv[0]))
    sys.exit(1)

## 0. Set torch configurations ################################################

torch.set_default_tensor_type(torch.FloatTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# If you use GPU, the device should be cuda
print('Device: {}'.format(device))

## 1. Load datasets and create data loaders ###################################

dataset_root = f'data/datasets/synthetic/{dataset_name}_0.2-0.2_'

# Datasets
train_dataset = [data.to(device) for data in
                 torch.load(f'{dataset_root}train.pt')]
val_dataset = [data.to(device) for data in torch.load(f'{dataset_root}val.pt')]
test_dataset = [data.to(device) for data in
                torch.load(f'{dataset_root}test.pt')]

# Data loaders
train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

## 2. Create the model ########################################################

layers = {'mlp': MLP, 'gcn': GCN, 'sage': GraphSAGE, 'gatv2': GATv2}
num_hidden_layers = num_layers - 2
is_gnn = layer_name in ['gcn', 'sage', 'gatv2']

model = layers[layer_name](input_dim=train_dataset[0].num_features,
                           hidden_dim=hidden_dim, output_dim=1,
                           num_hidden_layers=num_hidden_layers).to(device)

## 3. Train the model and save it #############################################

# Paths variables
scenario = dataset_name.split('_')[-1]
config = f'{layer_name}_{scenario}_{lr}_{num_layers}_{hidden_dim}_b{batch_size}'
training_results_path = f'data/tuning_results/{dataset_name}/{layer_name}/{config}.png'
best_model_path = f'models/tuning/{dataset_name}/{layer_name}/{config}.pt'

# Training parameters
epochs = 10000
early_stopping_steps = 1000
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam

# Train the model
torch.manual_seed(0)
model.reset_parameters()

# Save the model
best_model = train_model(original_model=model,
                         train_data_loader=train_data_loader,
                         val_data_loader=val_data_loader, optimizer=optimizer,
                         loss_fn=criterion, lr=lr, epochs=epochs,
                         early_stopping_steps=early_stopping_steps,
                         is_gnn=is_gnn, results_file=training_results_path,
                         model_file=best_model_path)

## 4. Print the best results ##################################################

print(
    f'\nTrain loss: {test_torch(best_model, train_data_loader, criterion, is_gnn):.4f}')
print(
    f'Validation loss: {test_torch(best_model, val_data_loader, criterion, is_gnn):.4f}')
print(
    f'Test loss: {test_torch(best_model, test_data_loader, criterion, is_gnn):.4f}')

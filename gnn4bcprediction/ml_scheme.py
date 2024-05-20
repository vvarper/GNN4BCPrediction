import copy
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt


@torch.no_grad()
def test_torch(model, data_loader, loss_fn, gnn=True):
    model.eval()

    total_loss = 0

    for batch in data_loader:
        if gnn:
            y_pred = torch.squeeze(model.forward(batch.x, batch.edge_index))
        else:
            y_pred = torch.squeeze(model.forward(batch.x))

        loss = loss_fn(y_pred, batch.y).item()
        total_loss += loss * batch.num_graphs

    total_loss = total_loss / len(data_loader.dataset)

    return total_loss


@torch.no_grad()
def test_sklearn(model, data_loader, loss_fn, gnn=True):
    model.eval()

    y_true = np.array([])
    y_pred = np.array([])

    for batch in data_loader:
        if gnn:
            new_pred = torch.squeeze(model.forward(batch.x, batch.edge_index))
        else:
            new_pred = torch.squeeze(model.forward(batch.x))

        y_true = np.append(y_true, batch.y.detach().cpu().numpy())
        y_pred = np.append(y_pred, new_pred.detach().cpu().numpy())

    return loss_fn(y_true, y_pred)


def train_epoch(model, train_data_loader, optimizer, loss_fn, gnn=True):
    total_loss = 0
    model.train()

    for batch in train_data_loader:
        optimizer.zero_grad()
        if gnn:
            out = torch.squeeze(model.forward(batch.x, batch.edge_index))
        else:
            out = torch.squeeze(model.forward(batch.x))

        loss = loss_fn(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    total_loss /= len(train_data_loader.dataset)

    return total_loss


def train_model(original_model, train_data_loader, val_data_loader, optimizer,
                loss_fn, lr, epochs, early_stopping_steps, is_gnn=True,
                results_file=None, model_file=None):
    model = copy.deepcopy(original_model)
    optimizer = optimizer(model.parameters(), lr=lr)
    best_model = None
    best_valid_loss = float('inf')
    no_improvement = 0

    train_losses, valid_losses = [], []

    initial_time = time.time()

    for epoch in range(epochs):
        loss = train_epoch(model, train_data_loader, optimizer, loss_fn,
                           is_gnn)
        train_loss = test_torch(model, train_data_loader, loss_fn, is_gnn)
        valid_loss = test_torch(model, val_data_loader, loss_fn, is_gnn)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            no_improvement = 0

        elif no_improvement > early_stopping_steps:
            print(f'Early stopping! (epochs: {epoch})')
            break
        else:
            no_improvement += 1

        if epoch % 100 == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {train_loss:.4f}, '
                  f'Valid: {valid_loss:.4f}')

    ending_time = time.time()

    print('Training time: ', ending_time - initial_time)

    if results_file is not None:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        save_training_results(train_losses, valid_losses, results_file)

    if model_file is not None:
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(best_model.state_dict(), model_file)
        print(f'Training ended for model {model_file}')

    return best_model


def save_training_results(train_losses, valid_losses, save_file):
    plt.clf()
    plt.plot([l for l in train_losses], label='Train')
    plt.plot([l for l in valid_losses], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot
    plt.savefig(save_file)

from typing import Tuple
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



def train_mlp(model, train_loader, num_epochs: int = 50, lr: float = 0.001, weight: float = 0.9) -> list[int]:
    """
    Trains a single MLP model using BCEWithLogitsLoss and Adam optimizer.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight (float): Positive class weight for loss function.
        
    Returns:
        list[float]: List of average loss values per epoch.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.train()
    loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        
        scheduler.step()
        current_loss = epoch_loss / len(train_loader)
        loss.append(current_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss[-1]:.4f}')

    return loss


def train_all_mlps(mlp: dict, train_loader: DataLoader, weight: float, n_epochs: int, lr: float) -> dict:
    """
    Trains multiple MLP models provided in a dictionary.
    
    Args:
        mlp (dict): Dictionary of {name: model}.
        train_loader (DataLoader): DataLoader for training data.
        weight (float): Positive class weight.
        n_epochs (int): Number of epochs.
        lr (float): Learning rate.
        
    Returns:
        dict: Dictionary of {name: loss_history_list}.
    """
    train_loss_list = {}
    for name, model in mlp.items():
        print(f"Training model: {name}")
        loss = train_mlp(model, train_loader, num_epochs=n_epochs, weight=weight, lr=lr)
        train_loss_list[name] = loss
    return train_loss_list

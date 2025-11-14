from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

def evaluate_mlp(model, data_loader, threshold: float = 0.5) -> dict:
    model.eval()
    probs_list = []
    y_list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            out = model(X_batch)
            out = out.view(-1)
            prob = torch.sigmoid(out).cpu().numpy()
            y_true = y_batch.view(-1).cpu().numpy()
            probs_list.append(prob)
            y_list.append(y_true)

    y_prob = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    print("balanced_accuracy:", round(metrics['balanced_accuracy'], 4))
    print("f1_macro:", round(metrics['f1_macro'], 4))
    print("roc_auc:", metrics['roc_auc'])
    print()
    return metrics


def evaluate_mlps(models: Dict[str, nn.Module], data_loader, threshold: float = 0.5):
    for name, model in models.items():
        print(f"--- {name} ---")
        evaluate_mlp(model, data_loader, threshold)
from src.data_and_preprocessors.preprocessors import build_transformer_for_regression
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def explain_mlp_model_with_shap(model: torch.nn.Module, 
                                X_train: object, 
                                X_test: object, 
                                numerical_cols: List[str], 
                                categorical_cols: List[str],
                                n_samples: int = 100):
    """
    Generates a SHAP summary plot for a PyTorch MLP model using DeepExplainer.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        X_train (pd.DataFrame): Training data (used for background distribution).
        X_test (pd.DataFrame): Test data (used for explanation).
        numerical_cols (List[str]): List of numerical column names.
        categorical_cols (List[str]): List of categorical column names.
        n_samples (int): Number of samples to use from train/test for SHAP (to speed up calculation).
    """
    transformer = build_transformer_for_regression(numerical_cols, categorical_cols)
    transformer.fit(X_train)

    X_train_sample = X_train.sample(n=min(n_samples, len(X_train)), random_state=42)
    X_test_sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=42)
    
    X_train_transformed = transformer.transform(X_train_sample)
    X_test_transformed = transformer.transform(X_test_sample)

    X_train_tensor = torch.FloatTensor(X_train_transformed)
    X_test_tensor = torch.FloatTensor(X_test_transformed)

    model.eval()
    explainer = shap.DeepExplainer(model, X_train_tensor)
    shap_values = explainer.shap_values(X_test_tensor)

    # 5. Plot summary
    # DeepExplainer может возвращать список массивов, даже если выход один
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Если shap_values имеет лишнюю размерность (например, (N, M, 1)), убираем её
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values.squeeze(2)
    
    feature_names = transformer.get_feature_names_out()
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
    plt.show()





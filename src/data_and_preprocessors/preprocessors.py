from typing import List
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, KBinsDiscretizer

def build_transformer_for_tree(categorical_cols: List[str]) -> ColumnTransformer:
    """
    Builds a ColumnTransformer for tree-based models.
    Applies log transformation to income and OneHotEncoding to categorical columns.
    
    Args:
        categorical_cols (List[str]): List of categorical column names.
        
    Returns:
        ColumnTransformer: Configured transformer.
    """
    return ColumnTransformer(
        transformers=[
            ('income_log', FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ['person_income']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],    
        remainder='passthrough'
        )


def build_transformer_for_regression(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Builds a ColumnTransformer for regression/linear models and MLP.
    Applies scaling, log transformation, binning for age, and OneHotEncoding.
    
    Args:
        numerical_cols (List[str]): List of numerical column names.
        categorical_cols (List[str]): List of categorical column names.
        
    Returns:
        ColumnTransformer: Configured transformer.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('income_log', FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ['person_income']),
            ('age_binned', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'), ['person_age']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],    
        remainder='drop'
    )
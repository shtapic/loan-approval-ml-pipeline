from typing import List, Dict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.data_and_preprocessors.preprocessors import build_transformer_for_tree, build_transformer_for_regression
import lightgbm as lgb
from lightgbm import LGBMClassifier


def make_pipelines(numerical: List[str], categorical: List[str], random_state: int) -> Dict[str, Pipeline]:
    """
    Creates a dictionary of sklearn Pipelines for different models (RF, LR, LGBM).
    Each pipeline includes the appropriate preprocessor and classifier.
    
    Args:
        numerical (List[str]): List of numerical column names.
        categorical (List[str]): List of categorical column names.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        Dict[str, Pipeline]: Dictionary with keys 'rf', 'lr', 'lgbm'.
    """
    preproc_for_tree = build_transformer_for_tree(categorical)
    preproc_for_regression = build_transformer_for_regression(numerical, categorical)

    pipes = {}
    pipes['rf'] = Pipeline([('pre', preproc_for_tree),
                            ('clf', RandomForestClassifier(random_state=random_state))])
    
    pipes['lr'] = Pipeline([('pre', preproc_for_regression),
                            ('clf', LogisticRegression(solver='saga', class_weight='balanced',
                                    random_state=random_state, max_iter=1000))])

    pipes['lgbm'] = Pipeline([('pre', preproc_for_regression),
                              ('clf', lgb.LGBMClassifier(random_state=random_state, 
                                    class_weight='balanced', n_jobs=-1))])
    
    return pipes

from typing import List, Dict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.data_and_preprocessors.preprocessors import build_transformer_for_tree, build_transformer_for_regression
import lightgbm as lgb
from lightgbm import LGBMClassifier


def make_pipelines(numerical: List[str], categorical: List[str], random_state: int) -> Dict[str, Pipeline]:
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

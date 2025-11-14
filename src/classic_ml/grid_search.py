from typing import Dict
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

def grid_search(pipes: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Dict[str, Pipeline]:
    best = {}
    # RF
    rf_cv = GridSearchCV(
        estimator=pipes['rf'],
        param_grid={'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 20, None], 'clf__min_samples_split': [2, 5]},
        scoring='balanced_accuracy', cv=5, n_jobs=-1, verbose=0
    )
    rf_cv.fit(X_train, y_train)
    best['rf'] = rf_cv.best_estimator_
    print("RF best:", rf_cv.best_params_)

    # LR
    lr_cv = GridSearchCV(
        estimator=pipes['lr'],
        param_grid={'clf__C': [0.1, 1, 10], 'clf__penalty': ['l1', 'l2'], 'clf__solver': ['saga', 'liblinear']},
        scoring='balanced_accuracy', cv=5, n_jobs=-1, verbose=0
    )
    lr_cv.fit(X_train, y_train)
    best['lr'] = lr_cv.best_estimator_
    print("LR best:", lr_cv.best_params_)

    # LGBM
    lgb_cv = RandomizedSearchCV(
        estimator=pipes['lgbm'],
        param_distributions={'clf__num_leaves': [31, 60], 'clf__learning_rate': [0.05, 0.1], 'clf__n_estimators': [100, 200, 300]},
        n_iter=20, scoring='balanced_accuracy', cv=5, n_jobs=-1, verbose=1, random_state=random_state
    )
    lgb_cv.fit(X_train, y_train)
    best['lgbm'] = lgb_cv.best_estimator_
    print("LGBM best:", lgb_cv.best_params_)

    return best
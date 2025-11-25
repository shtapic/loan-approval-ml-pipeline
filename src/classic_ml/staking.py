from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def build_and_fit_stacking(rf_model, lgbm_model, X_train, y_train, random_state):
    """
    Builds and trains a StackingClassifier using Random Forest and LightGBM as base learners
    and Logistic Regression as the final meta-learner.
    
    Args:
        rf_model: Trained Random Forest model (or pipeline).
        lgbm_model: Trained LightGBM model (or pipeline).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int): Random seed.
        
    Returns:
        StackingClassifier: Trained stacking model.
    """
    final = LogisticRegression(solver='saga', class_weight='balanced', random_state=random_state, max_iter=1000)
    stack = StackingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lgbm', lgbm_model)
        ],
        final_estimator=final,
        n_jobs=-1,
        passthrough=False,
        cv=5
    )
    stack.fit(X_train, y_train)
    return stack
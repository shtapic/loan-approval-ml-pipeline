from typing import Dict
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

def evaluate_pipes(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> None:
    for name, m in models.items():
        y_pred = m.predict(X_test)
        try:
            y_prob = m.predict_proba(X_test)
            roc = roc_auc_score(y_test, y_prob[:,1])
        except Exception:
            roc = None
        print(f"--- {name} ---")
        print("balanced_accuracy:", balanced_accuracy_score(y_test, y_pred))
        print("f1_macro:", f1_score(y_test, y_pred, average='macro'))
        print("roc_auc_macro:", roc)
        print()
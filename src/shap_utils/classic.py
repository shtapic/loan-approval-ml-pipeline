import shap
import matplotlib.pyplot as plt

def explain_classic_model_with_shap(pipeline, x_test):
    """
    Generates a SHAP summary plot for a classic ML pipeline (sklearn-compatible).
    
    Handles both Tree-based models (RandomForest, LGBM, CatBoost) and Linear models.
    Automatically extracts the classifier and preprocessor from the pipeline.
    
    Args:
        pipeline (Pipeline): Fitted sklearn Pipeline containing 'pre' (preprocessor) and 'clf' (classifier).
        x_test (pd.DataFrame): Test data (features only).
    """
    model = pipeline.named_steps['clf']
    preprocessor = pipeline.named_steps['pre']
    
    x_test_transformed = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()

    if model.__class__.__name__ in ['RandomForestClassifier', 'LGBMClassifier', 'CatBoostClassifier']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test_transformed)
        
        # Для бинарной классификации берем значения для класса 1
        # RandomForestClassifier возвращает список [class_0, class_1]
        # LGBMClassifier может возвращать один массив или список
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        # Если shap_values имеет размерность (n_samples, n_features, 2) - это тоже случай RF в некоторых версиях
        elif len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
             shap_values = shap_values[:, :, 1]
    else:
        explainer = shap.LinearExplainer(model, x_test_transformed)
        shap_values = explainer.shap_values(x_test_transformed)

    shap.summary_plot(shap_values, x_test_transformed, feature_names=feature_names)
 


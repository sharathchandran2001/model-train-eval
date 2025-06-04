from metrics import performance, robustness, explainability, fairness #, drift
import pandas as pd






def run_evaluation(model, X_train, X_test, y_train, y_test, sensitive_features=None):
    results = {}

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except:
        y_proba = None

    results["performance"] = performance.evaluate_performance(y_test, y_pred, y_proba)
    results["robustness"] = robustness.evaluate_robustness(model, X_test, y_test)
    results["shap_values"] = explainability.compute_shap_summary(model, X_test)
    # Optional: evaluate data drift
    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)
#     results["data_drift"] = drift.evaluate_data_drift(train_df, test_df)

    if sensitive_features is not None:
        results["fairness"] = fairness.evaluate_fairness(y_test, y_pred, sensitive_features)

    return results
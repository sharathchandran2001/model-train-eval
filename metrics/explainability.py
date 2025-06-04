import shap

def compute_shap_summary(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values

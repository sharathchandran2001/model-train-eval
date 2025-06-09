import numpy as np
from sklearn.metrics import accuracy_score

def compare_predictions(model, X_recent, y_actual, y_previous_preds=None):
    y_pred = model.predict(X_recent)
    result = {
        "current_accuracy": accuracy_score(y_actual, y_pred)
    }
    if y_previous_preds is not None:
        drift_rate = np.mean(y_pred != y_previous_preds)
        result["prediction_drift_rate"] = drift_rate
    return result

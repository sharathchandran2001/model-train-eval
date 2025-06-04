from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

def evaluate_fairness(y_true, y_pred, sensitive_features):
    frame = MetricFrame(metrics=selection_rate,
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=sensitive_features)
    return {
        "selection_rate_by_group": frame.by_group,
        "demographic_parity_diff": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    }

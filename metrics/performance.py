from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def evaluate_performance(y_true, y_pred, y_proba=None):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    if y_proba is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr')
    return results

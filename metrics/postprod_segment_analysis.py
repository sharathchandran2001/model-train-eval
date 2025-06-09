import pandas as pd
from sklearn.metrics import f1_score

def segment_metrics(df, model, segment_col, label_col):
    results = {}
    for segment_value in df[segment_col].unique():
        subset = df[df[segment_col] == segment_value]
        X = subset.drop(columns=[label_col, segment_col])
        y = subset[label_col]
        y_pred = model.predict(X)
        results[segment_value] = {
            "f1_score": f1_score(y, y_pred, average='weighted')
        }
    return results

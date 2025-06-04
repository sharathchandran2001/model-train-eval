from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def evaluate_data_drift(train_df, test_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train_df, current_data=test_df)
    result = report.as_dict()

    drift_score = result['metrics'][0]['result']['dataset_drift']
    return {
        "data_drift_detected": drift_score,
        "drift_details": result['metrics'][0]['result']
    }

from metrics import postprod_prediction_monitor, postprod_segment_analysis

def run_postprod_checks(model, X_live, y_live, previous_preds=None, df_segmented=None):
    results = {}

    # Monitor accuracy drift
    results["prediction_monitoring"] = postprod_prediction_monitor.compare_predictions(
        model, X_live, y_live, y_previous_preds=previous_preds)

    # Segment metrics
    if df_segmented is not None:
        results["segment_analysis"] = postprod_segment_analysis.segment_metrics(
            df_segmented, model, segment_col="region", label_col="target")

    return results

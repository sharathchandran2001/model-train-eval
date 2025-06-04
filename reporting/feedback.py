def generate_feedback(performance_metrics):
    feedback = []

    if performance_metrics["accuracy"] < 0.8:
        feedback.append("⚠️ Accuracy is below acceptable threshold. Review feature engineering or model choice.")
    if performance_metrics["f1_score"] < 0.75:
        feedback.append("⚠️ F1-score is low. Model may be struggling with class imbalance or recall.")
    if performance_metrics["precision"] < 0.75:
        feedback.append("🔎 Low precision may indicate many false positives. Examine training labels or regularization.")
    if performance_metrics["recall"] < 0.75:
        feedback.append("🧐 Low recall may mean the model misses too many true cases. Try rebalancing or reducing bias.")

    if not feedback:
        feedback.append("✅ Model meets all KPI thresholds. Good job!")

    return feedback

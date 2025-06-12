import joblib
from pipeline import run_evaluation
from reporting import report_generator, feedback

# Load model and data
model = joblib.load("data/model.joblib")
X_train, X_test, y_train, y_test = joblib.load("data/data_splits.joblib")

# Run evaluation
results = run_evaluation(model, X_train, X_test, y_train, y_test)

# Display summary in terminal
for section, values in results.items():
    if section != "shap_values":
        print(f"=== {section.upper()} ===")
        print(values)

# Developer feedback
dev_feedback = feedback.generate_feedback(results["performance"])
print("\n Developer Feedback:")
for note in dev_feedback:
    print(note)

# Save reports
report_generator.generate_text_report(results, "model_report.txt")
report_generator.generate_json_report(results, "model_report.json")
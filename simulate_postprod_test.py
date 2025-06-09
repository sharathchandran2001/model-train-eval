import joblib
import pandas as pd
from postprod_pipeline import run_postprod_checks

# Load model and data
model = joblib.load("data/model.joblib")
X_train, X_test, y_train, y_test = joblib.load("data/data_splits.joblib")

# Simulate live data (slice of test set)
X_live = X_test[:20]
y_live = y_test[:20]
previous_preds = model.predict(X_live)

# Simulate a segment column (e.g., by creating dummy regions)
df_live = pd.DataFrame(X_live)
df_live["target"] = y_live
df_live["region"] = ["North", "South", "East", "West", "North"] * 4  # Simulated segments

# Run post-production checks
results = run_postprod_checks(model, X_live, y_live, previous_preds, df_segmented=df_live)

# Print results
print("=== Post-Production Evaluation Results ===")
for section, metrics in results.items():
    print(f"\n--- {section.upper()} ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

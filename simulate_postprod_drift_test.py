from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from postprod_pipeline import run_postprod_checks

# Load a trained model and original predictions
model = joblib.load("data/model.joblib")
X_train, X_test, y_train, y_test = joblib.load("data/data_splits.joblib")
previous_preds = model.predict(X_test[:20])  # Use a stable baseline for drift comparison

# Load mismatched dataset (Wine dataset)
wine = load_wine()
# X_fake = wine.data[:20]
X_fake = wine.data[:20, :4]
y_fake = wine.target[:20]

# Create dummy region segments for fake data
df_fake = pd.DataFrame(X_fake)
df_fake["target"] = y_fake
df_fake["region"] = ["North", "South", "East", "West", "North"] * 4

# Run post-production evaluation using mismatched live data
results = run_postprod_checks(model, X_fake, y_fake, previous_preds, df_segmented=df_fake)

# Display results
print("=== Post-Production Evaluation with Mismatched Data ===")
for section, metrics in results.items():
    print(f"\n--- {section.upper()} ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")
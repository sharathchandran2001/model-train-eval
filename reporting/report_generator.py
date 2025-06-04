import json
from datetime import datetime

def generate_text_report(results, filename="model_report.txt"):
    with open(filename, "w") as f:
        f.write(f"Model Evaluation Report\nGenerated: {datetime.now()}\n\n")
        for section, metrics in results.items():
            if section == "shap_values":
                continue  # Skip SHAP raw values in text
            f.write(f"--- {section.upper()} ---\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

def generate_json_report(results, filename="model_report.json"):
    filtered = {k: v for k, v in results.items() if k != "shap_values"}
    with open(filename, "w") as f:
        json.dump(filtered, f, indent=4)

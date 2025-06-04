from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save_model(model_path="model.joblib", data_path="data"):
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(data_path, exist_ok=True)
    joblib.dump(model, os.path.join(data_path, model_path))
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(data_path, "data_splits.joblib"))

    print(f"Model and data saved in '{data_path}' directory.")

if __name__ == "__main__":
    train_and_save_model()
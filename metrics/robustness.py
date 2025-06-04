import numpy as np
from sklearn.metrics import accuracy_score

def add_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def evaluate_robustness(model, X, y):
    X_noisy = add_noise(X)
    y_pred_noisy = model.predict(X_noisy)
    return {
        "robust_accuracy": accuracy_score(y, y_pred_noisy)
    }

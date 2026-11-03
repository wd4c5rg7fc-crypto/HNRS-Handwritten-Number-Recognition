"""Model loading and prediction functions for HNRS."""

from pathlib import Path
import numpy as np


def load_trained_model(kind="CNN"):
    import tensorflow as tf

    kind = kind.upper()

    # Select the appropriate trained model
    if kind == "CNN":
        # Best CNN trained and evaluated during Week 7
        path = Path("models/best_cnn_model.keras")

    elif kind == "MLP":
        # MLP model trained by the team
        path = Path("models/mlp.keras")

    else:
        raise ValueError("Model must be CNN or MLP")

    # Check whether the model exists
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Load and return the trained model
    return tf.keras.models.load_model(path)


def predict_digits(model, digit_tensors):
    """Predict digits from a list of 28x28x1 images."""

    if not digit_tensors:
        return []

    # Convert digit images into a NumPy batch
    batch = np.stack(digit_tensors).astype("float32")

    # Verify input dimensions
    if batch.ndim != 4 or batch.shape[1:] != (28, 28, 1):
        raise ValueError("Expected batch of 28x28x1 digits")

    # Predict digit probabilities
    probs = model.predict(batch, verbose=0)

    # Return predicted digit and confidence
    return [
        (int(row.argmax()), float(row.max()))
        for row in probs
    ]
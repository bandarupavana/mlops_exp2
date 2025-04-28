import pickle
import numpy as np

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl not found!")
    exit(1)

# Example new input values for prediction (one row per patient)
new_data = np.array([
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],  # Example 1
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],   # Example 2
    [8, 183, 64, 0, 0, 23.3, 0.672, 32]    # Example 3
])

# Ensure input shape is correct
if hasattr(model, "predict"):
    try:
        predictions = model.predict(new_data)
        predicted_labels = ["Diabetes" if pred == 1 else "No Diabetes" for pred in predictions]

        # Display predictions
        for i, label in enumerate(predicted_labels):
            print(f"Patient {i+1}: {label}")
    except Exception as e:
        print(f"Prediction error: {e}")
else:
    print("Error: Loaded model does not support `predict` method.")

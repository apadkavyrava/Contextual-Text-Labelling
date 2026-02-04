"""
Text classification - interactive mode.
Enter a word or consposite noun and get parsing of meaning
"""

import joblib

CONFIDENCE_THRESHOLD = 0.7  # Adjust this value as needed (0.0 to 1.0)


def load_model(model_path="model.joblib"):
    """Load the trained pipeline."""
    return joblib.load(model_path)


def predict(text, model, threshold=CONFIDENCE_THRESHOLD):
    """
    Predict category for input text.
    Returns "Others" if confidence is below threshold.
    """
    proba = model.predict_proba([text])[0]
    max_proba = max(proba)
    
    if max_proba < threshold:
        return "Others"
    
    return model.predict([text])[0]


def main():
    model = load_model("model.joblib")
    
    text = input("Enter text: ")
    result = predict(text, model)
    print(f"Category: {result}")


if __name__ == "__main__":
    main()
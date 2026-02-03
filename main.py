"""
Simple text classification inference script.
Works with a sklearn Pipeline that includes TfidfVectorizer + LogisticRegression.
"""

import joblib


def load_model(model_path="model.joblib"):
    """Load the trained pipeline."""
    model = joblib.load(model_path)
    return model


def predict(text, model):
    """
    Return model prediction for input text.
    
    Args:
        text: Input text string (e.g., "Fish & Son ltd", "Green apple")
        model: Trained sklearn Pipeline (TfidfVectorizer + LogisticRegression)
    
    Returns:
        prediction: Model prediction
        probabilities: Prediction probabilities (if available)
    """
    # Pipeline handles vectorization internally - just pass raw text
    prediction = model.predict([text])[0]
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
    
    return prediction, probabilities


def main():
    # Load the pipeline
    model = load_model()
    
    # Example usage
    test_texts = [
        "Fish & Son ltd",
        "Green apple",
        "New York"
    ]
    
    print("Text Classification Results")
    print("-" * 40)
    
    for text in test_texts:
        prediction, probs = predict(text, model)
        print(f"Input: {text}")
        print(f"Prediction: {prediction}")
        if probs is not None:
            print(f"Probabilities: {probs}")
        print()


if __name__ == "__main__":
    main()
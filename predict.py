import pickle

def predict_text(model, vectorizer, text):
    # Load the model
    with open(model, "rb") as f:
        model = pickle.load(f)

    # Load the vectorizer
    with open(vectorizer, "rb") as f:
        _, _, vectorizer = pickle.load(f)

    # Vectorize the input email
    text_vector = vectorizer.transform([text])

    # Make a prediction
    prediction = model.predict(text_vector)
    return "Phishing" if prediction[0] == 1 else "Not Phishing"

if __name__ == "__main__":
    text = "Congratulations! You have won a $1,000 gift card. Click here to claim your prize immediately: http://fake-link.com. Act fast, offer expires today!"
    result = predict_text("phishing_detector.pkl", "preprocessed_data.pkl", text)
    print(f"Prediction: {result}")


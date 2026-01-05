import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def data_preprocess(input, output):
    data = pd.read_csv(input, encoding="ISO-8859-1")

    # label encoding
    data["v1"] = data["v1"].map({"ham": 0, "spam": 1})
    # rename columns
    data = data.rename(columns={"v1": "label", "v2": "text"})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["text"])
    y = data["label"]

    with open(output, "wb") as f:
        pickle.dump((X, y, vectorizer), f)

    print(f"Preprocessed data saved to {output}")

if __name__ == "__main__":
    data_preprocess("spam.csv", "preprocessed_data.pkl")

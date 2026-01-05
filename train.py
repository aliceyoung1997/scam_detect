import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
#from sklearn.metrics import accuracy_score

def train_model(data_file, model_file):
    # Load preprocessed data
    with open(data_file, "rb") as f:
        X, y, _ = pickle.load(f)
        y = y.copy() # to solve the ValueError: cannot set WRITEABLE flag to True of this array

    # Train a Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Stratified for imbalanced data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print("Mean CV Accuracy:", scores.mean())
    
    # fit model on full datase
    model.fit(X, y)

    # Save the trained model
    
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_model("preprocessed_data.pkl", "phishing_detector.pkl")

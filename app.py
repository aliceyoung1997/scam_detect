
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_text

class Text(BaseModel):
    text: str

app = FastAPI()
@app.post("/predict")
def predict(input: Text):
    #result = predict_text("phishing_detector.pkl", "preprocessed_data.pkl", input.text)
    #return {"prediction": result}
    try:
        result = predict_text("phishing_detector.pkl", "preprocessed_data.pkl", input.text)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


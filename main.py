from fastapi import FastAPI
from schema import NewsRequest

import joblib

app = FastAPI(
    title="News Sentiment API"
)

# Load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {
        "message": "News Sentiment Analysis API"
    }

@app.post("/predict")
def predict(request: NewsRequest):

    prediction = model.predict([request.text])[0]

    probability = model.predict_proba([request.text])[0]

    return {
        "text": request.text,
        "prediction": prediction,
        "confidence": {
            "negative": float(probability[0]),
            "neutral": float(probability[1]),
            "positive": float(probability[2])
        }
    }
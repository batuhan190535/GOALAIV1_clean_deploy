from fastapi import FastAPI
from utils.predict import predict_match
import joblib

app = FastAPI()
model = joblib.load("models/ms_model.pkl")

@app.get("/")
def home():
    return {"status": "GOALMASTER LITE is alive!"}

@app.get("/predict")
def predict(home_form: float, away_form: float, home_goals_avg: float, away_goals_avg: float, goal_diff: float):
    features = [[home_form, away_form, home_goals_avg, away_goals_avg, goal_diff]]
    prediction = model.predict(features)[0]
    return {"prediction": f"Match Result Prediction: {prediction}"}
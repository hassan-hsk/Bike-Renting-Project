from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
from backend.model_loader import model_scores
from backend.schemas import PredictionInput



app = FastAPI()

class PredictionInput(BaseModel):
    season: int
    yr: int
    mnth: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float
    day: int
    dayofweek: int
    model: str

@app.post("/predict")
def predict_bike_demand(data: PredictionInput):
    model = model_scores.get(data.model)
    if not model:
        return {"error": "Invalid model name"}

    df = pd.DataFrame([[data.season, data.yr, data.mnth, data.holiday, data.weekday,
                        data.workingday, data.weathersit, data.temp, data.atemp,
                        data.hum, data.windspeed, data.day, data.dayofweek]],
                      columns=["season", "yr", "mnth", "holiday", "weekday", "workingday",
                               "weathersit", "temp", "atemp", "hum", "windspeed", "day", "dayofweek"])
    prediction = model.predict(df)[0]
    return {"prediction": int(round(prediction))}

@app.get("/scores")
def get_model_scores():
    with open("backend/models/r2_scores.json") as f:
        raw_scores = json.load(f)

    return {
        "Random Forest": round(raw_scores.get("random_forest", 0), 4),
        "Decision Tree": round(raw_scores.get("decision_tree", 0), 4),
        "Linear Regression": round(raw_scores.get("linear_regression", 0), 4),
    }

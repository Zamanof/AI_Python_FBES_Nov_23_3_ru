from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import io

import numpy as np

app = FastAPI(title="Price Predictions AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEl_PATH = Path(__file__).parent / "models" / "model_az.pkl"

try:
    loaded = joblib.load(MODEl_PATH)
    if isinstance(loaded, dict) and "pipline" in loaded:
        BUNDLE = loaded
        PIPELINE = BUNDLE["pipline"]
        FEATURES = BUNDLE.get("feature_order", ["Bedrooms", "Bathrooms", "Sqm", "City"])
    print(f"Model loaded successfully from {MODEl_PATH}")

except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    BUNDLE = None
    PIPELINE = None
    FEATURES = ["Bedrooms", "Bathrooms", "Sqm", "City"]


class PredictIn(BaseModel):
    bedrooms: float
    bathrooms: float
    sqm: float
    city: str

@app.post("/predict")
async def predict(request: PredictIn):
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        feature_dict = {
            'Bedrooms': [request.bedrooms],
            'Bathrooms': [request.bathrooms],
            'City': [request.city]
        }
        if "Sqm" in FEATURES:
            feature_dict['Sqm'] = [request.sqm]
        features_df = pd.DataFrame(feature_dict, columns=FEATURES)
        prediction = PIPELINE.predict(features_df)[0]
        return {"priceAZN": float(prediction)}
    except Exception as e:
        print(f"Error predicting price: {e}")
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=f"Error predicting price: {e}")

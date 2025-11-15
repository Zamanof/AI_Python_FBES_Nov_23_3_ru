from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import io

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
    print(PIPELINE)
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



@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    global BUNDLE, PIPELINE, FEATURES

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_columns = ["PriceAZN","Bedrooms", "Bathrooms", "Sqm", "City"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400,
                                detail=f"Missing columns: {missing_columns}"
                                )

        X = df[["Bedrooms", "Bathrooms", "Sqm", "City"]]
        y = df["PriceAZN"].astype(float)

        # preprocessing
        numeric_features = ['Bedrooms', 'Bathrooms', 'Sqm']
        numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
        )

        categorical_features = ["City"]
        categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
        )

        preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
        )

        # model

        rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
        )

        # pipline
        pipeline = Pipeline(
        steps=[
            ('prep', preprocessor),
            ('model', rf)
        ]
        )

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        FEATURES = ["Bedrooms", "Bathrooms", "Sqm", "City"]

        BUNDLE = {
            "pipeline" : pipeline,
            "feature_order": FEATURES
        }

        MODEl_PATH.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(BUNDLE, MODEl_PATH)

        PIPELINE = pipeline

        return {
            "status": "success",
            "message": "Model trained successfully",
            "metrics": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 4),
            },
            "dataset_size": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retraining error: {e}")


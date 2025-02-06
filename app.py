from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
import os
from io import BytesIO

app = FastAPI()

MODEL_PATH = "models"

# Load models
with open(os.path.join(MODEL_PATH, 'random_forest.pkl'), 'rb') as f:
    rf_model = pickle.load(f)
with open(os.path.join(MODEL_PATH, 'gradient_boosting.pkl'), 'rb') as f:
    gb_model = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(BytesIO(await file.read()))

    rf_pred = rf_model.predict(df)
    gb_pred = gb_model.predict(df)

    predictions = {
        "RandomForest Prediction": rf_pred.tolist(),
        "GradientBoosting Prediction": gb_pred.tolist()
    }

    return predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

import os
import mlflow
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Global dictionary to store our loaded model and preprocessor
model_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This logic runs once when the server starts.
    It pulls the '@champion' model from MLflow.
    """
    try:

        # 1. Force the Tracking URI to the SQLite DB in the root
        # Get the directory where main.py lives (api/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to find the root folder (ml-risk-prediction/)
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        
        db_path = f"sqlite:///{project_root}/mlflow.db"
        mlflow.set_tracking_uri(db_path)

        # 1. Load the Preprocessor (Saved in Step 3)
        model_assets["preprocessor"] = joblib.load(os.path.join(project_root, "docker/preprocessor.joblib"))
        
        # 2. Load the Model from MLflow Registry using the Alias
        model_name = "ChurnPredictionXGB"
        model_uri = f"models:/{model_name}@champion"
        model_assets["model"] = mlflow.xgboost.load_model(model_uri)
        
        print(f"Successfully loaded {model_name}@champion")
    except Exception as e:
        print(f"Error during startup: {e}")
    yield
    # Clean up on shutdown if needed
    model_assets.clear()

app = FastAPI(title="Customer Churn Prediction Service", lifespan=lifespan)

# Define the input schema using Pydantic
class CustomerInput(BaseModel):
    age: int
    tenure: int
    usage_frequency: float
    support_calls: int
    payment_delay: int
    total_spend: float
    last_interaction: int
    gender: str
    subscription_type: str
    contract_length: str
    
@app.get("/health")
def health():
    return {"status": "online", "model": "ChurnPredictionXGB", "alias": "champion"}

@app.post("/predict")
async def predict(data: CustomerInput):
    if "model" not in model_assets:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        print(input_df)
        
        # Preprocess the data
        processed_data = model_assets["preprocessor"].transform(input_df)
        
        # Get Prediction
        prediction = model_assets["model"].predict(processed_data)
        probability = model_assets["model"].predict_proba(processed_data)[:, 1]
        
        return {
            "churn_risk": int(prediction[0]),
            "probability": round(float(probability[0]), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
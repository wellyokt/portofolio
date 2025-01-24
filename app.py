from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('api')

class FeatureInput(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: int
    Area_Income: float
    Daily_Internet_Usage: float
    Male: int
    Hour: int
    DayOfWeek: float
    
    class Config:
        schema_extra = {
            "example": {
                "Daily_Time_Spent_on_Site":68.95,
                "Age":35,
                "Area_Income":61833.9,
                "Daily_Internet_Usage":256.09,
                "Male":0,
                "Hour":0,
                "DayOfWeek":6
            }
        }

app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

#--- Jika deploy dengan docker aktifkan Cors -----
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Load model and scaler at startup
try:
    with open(Config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(Config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

@app.post("/predict")
async def predict(features: FeatureInput):
    # try:
        # Validate input
        for feature, value in features.dict().items():
            if not Config.is_valid_feature_value(feature, value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {feature}"
                )
        
        # Prepare input
        feature_dict = features.dict()
        input_df = pd.DataFrame([feature_dict])[Config.FEATURE_COLUMNS]
        # Scale features
        col_num =  ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage','Hour',  'DayOfWeek']
        input_df[col_num] = scaler.transform(input_df[col_num])

        # Make prediction
        prediction = model.predict(input_df)
        final_prediction = prediction[0]
        
        logger.info(f"Prediction made for input: {feature_dict}")
        return {"prediction": int(final_prediction)}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
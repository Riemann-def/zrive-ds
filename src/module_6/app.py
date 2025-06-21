# app.py
import uvicorn
from fastapi import FastAPI, HTTPException
from datetime import datetime
import time

from models.request import PredictRequest
from models.response import PredictResponse, StatusResponse
from exceptions.custom_exceptions import InvalidRequestException, ServiceUnavailableException
from services.prediction_service import PredictionService
from logger import ServiceLogger, ModelLogger
from config import Config

Config.create_directories()

service_logger = ServiceLogger()
model_logger = ModelLogger()

app = FastAPI(title=Config.API_TITLE, version=Config.API_VERSION)

try:
    prediction_service = PredictionService()
    service_logger.log_info("API initialized successfully")
except Exception as e:
    service_logger.log_error(f"Failed to initialize API: {str(e)}")
    raise ServiceUnavailableException("Service initialization failed")

@app.get("/status", response_model=StatusResponse)
def get_status():
    start_time = time.time()
    try:
        response = StatusResponse(status="OK", timestamp=datetime.now())
        latency = time.time() - start_time
        service_logger.log_request("GET", "/status", latency, "200")
        return response
    except Exception as e:
        latency = time.time() - start_time
        service_logger.log_request("GET", "/status", latency, "500")
        service_logger.log_error(f"Status endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()
    
    try:
        if not request.user_id or request.user_id.strip() == "":
            raise InvalidRequestException("user_id cannot be empty")
        
        prediction = prediction_service.predict(request.user_id)
        
        response = PredictResponse(
            user_id=request.user_id,
            predicted_price=prediction,
            timestamp=datetime.now()
        )
        
        latency = time.time() - start_time
        service_logger.log_request("POST", "/predict", latency, "200")
        model_logger.log_prediction(request.user_id, prediction)
        
        return response
        
    except InvalidRequestException as e:
        latency = time.time() - start_time
        service_logger.log_request("POST", "/predict", latency, "400")
        service_logger.log_error(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        latency = time.time() - start_time
        service_logger.log_request("POST", "/predict", latency, "500")
        service_logger.log_error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction service error")

if __name__ == "__main__":
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)

# config.py
import os
from pathlib import Path
from constants import DataFiles

class Config:
    MODULE_6_DIR = Path(__file__).parent

    LOGS_DIR = MODULE_6_DIR / "logs"
    SERVICE_LOG_FILE = LOGS_DIR / "service_metrics.txt"
    MODEL_LOG_FILE = LOGS_DIR / "model_metrics.txt"
    
    DATA_DIR = MODULE_6_DIR / "data"
    BIN_DIR = MODULE_6_DIR / "bin"
    MODEL_PATH = BIN_DIR / DataFiles.MODEL
    
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_TITLE = "Basket Prediction API"
    API_VERSION = "1.0.0"
    
    @classmethod
    def create_directories(cls):
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.BIN_DIR.mkdir(parents=True, exist_ok=True)
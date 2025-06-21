# logger.py
import os
from datetime import datetime
from pathlib import Path
from config import Config

class ServiceLogger:
    def __init__(self, log_file=None):
        if log_file is None:
            log_file = Config.SERVICE_LOG_FILE
        
        self.log_file = Path(log_file)
        
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write("SERVICE METRICS LOG\n")
                f.write("=" * 50 + "\n")
    
    def log_request(self, method: str, endpoint: str, latency: float, status_code: str):
        """HTTP request logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | REQUEST | {method} {endpoint} | latency={latency:.3f}s | status={status_code}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def log_error(self, error_message: str):
        """Service error logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | ERROR | {error_message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def log_info(self, info_message: str):
        """General information logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | INFO | {info_message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

class ModelLogger:
    def __init__(self, log_file=None):
        if log_file is None:
            log_file = Config.MODEL_LOG_FILE
            
        self.log_file = Path(log_file)
        
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write("MODEL METRICS LOG\n")
                f.write("=" * 50 + "\n")
    
    def log_prediction(self, user_id: str, prediction: float):
        """Model prediction logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | PREDICTION | user_id={user_id} | prediction={prediction:.2f}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def log_features(self, user_id: str, features: dict):
        """Model features logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        features_str = " | ".join([f"{k}={v}" for k, v in features.items()])
        log_entry = f"{timestamp} | FEATURES | user_id={user_id} | {features_str}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
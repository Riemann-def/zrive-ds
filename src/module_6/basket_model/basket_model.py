import os
import joblib
import numpy as np
from pathlib import Path
from exceptions.custom_exceptions import PredictionException
from config import Config


class BasketModel:
    def __init__(self):
        model_path = Config.MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, features: np.ndarray) -> np.ndarray:
        try:
            print(f"Predicting with features.shape {features.shape}: {features}")
            pred = self.model.predict(features)
        except Exception as e:
            raise PredictionException(f"Error during model inference: {str(e)}")
        return pred

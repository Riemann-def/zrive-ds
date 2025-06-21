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
            pred = self.model.predict(features)
        except Exception as exception:
            raise PredictionException("Error during model inference") from exception
        return pred

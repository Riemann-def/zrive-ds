# services/prediction_service.py
import numpy as np
from basket_model.basket_model import BasketModel
from basket_model.feature_store import FeatureStore
from exceptions.custom_exceptions import UserNotFoundException, PredictionException

class PredictionService:
    def __init__(self):
        self.model = BasketModel()
        self.feature_store = FeatureStore()
    
    def predict(self, user_id: str) -> float:
        try:
            features = self.feature_store.get_features(user_id)
            features_array = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features_array)
            return float(prediction[0])
            
        except UserNotFoundException:
            raise
        except PredictionException:
            raise
        except Exception as e:
            raise PredictionException(f"Unexpected error during prediction: {str(e)}")

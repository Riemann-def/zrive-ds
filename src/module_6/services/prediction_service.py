# services/prediction_service.py
import numpy as np
import time
from basket_model.basket_model import BasketModel
from basket_model.feature_store import FeatureStore
from exceptions.custom_exceptions import UserNotFoundException, PredictionException
from constants import FeatureColumns

class PredictionService:
    def __init__(self, model_logger):
        self.model = BasketModel()
        self.feature_store = FeatureStore()
        self.model_logger = model_logger
        self.feature_names = FeatureColumns.get_predictive_features()
    
    def predict(self, user_id: str) -> float:
        """
        :raises UserNotFoundException: If the user ID is not found in the feature store.
        :raises PredictionException: If there is an error during prediction.
        :raises Exception: For any unexpected errors during prediction.
        """
        start_time = time.time()
        try:
            # Obtener y procesar features
            raw_features = self.feature_store.get_features(user_id)
            last_order_features = raw_features.iloc[-1]
            features_array = np.array(last_order_features).reshape(1, -1)
            
            # Realizar predicción
            prediction = self.model.predict(features_array)
            prediction_value = float(prediction[0])
            
            # Log completo en una sola llamada
            latency = time.time() - start_time
            if self.model_logger:
                self.model_logger.log_complete_prediction(
                    user_id=user_id,
                    prediction=prediction_value,
                    latency=latency,
                    raw_features=raw_features,
                    selected_features=last_order_features,
                    feature_names=self.feature_names,
                    input_features=features_array
                )
            
            return prediction_value
            
        except UserNotFoundException as e:
            if self.model_logger:
                self.model_logger.log_error(user_id, "UserNotFound", str(e))
            raise
        except PredictionException as e:
            if self.model_logger:
                self.model_logger.log_error(user_id, "PredictionError", str(e))
            raise
        except Exception as e:
            if self.model_logger:
                self.model_logger.log_error(user_id, "UnexpectedError", str(e))
            raise PredictionException(f"Unexpected error during prediction: {str(e)}")
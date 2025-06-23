# tests/module_7/test_prediction_service.py
import pytest
import numpy as np

from src.module_7.services.prediction_service import PredictionService
from exceptions.custom_exceptions import (
    UserNotFoundException,
    PredictionException,
)


class TestPredictionService:
    def test_predict_success(self, mock_model, mock_feature_store, mock_model_logger):
        """Test successful prediction"""
        service = PredictionService(mock_model_logger)
        service.model = mock_model
        service.feature_store = mock_feature_store

        user_id = "test_user_123"
        expected_prediction = 75.42

        result = service.predict(user_id)

        assert isinstance(result, float)
        assert result == expected_prediction
        mock_feature_store.get_features.assert_called_once_with(user_id)
        mock_model.predict.assert_called_once()

        model_input = mock_model.predict.call_args[0][0]
        assert model_input.shape == (1, 4)

    def test_predict_user_not_found(
        self, mock_model, mock_feature_store, mock_model_logger
    ):
        """Test prediction when user doesn't exist"""
        service = PredictionService(mock_model_logger)
        service.model = mock_model
        service.feature_store = mock_feature_store

        mock_feature_store.get_features.side_effect = UserNotFoundException(
            "User not found"
        )

        with pytest.raises(UserNotFoundException):
            service.predict("nonexistent_user")

    def test_predict_model_error(
        self, mock_model, mock_feature_store, mock_model_logger
    ):
        """Test prediction when model fails"""
        service = PredictionService(mock_model_logger)
        service.model = mock_model
        service.feature_store = mock_feature_store

        mock_model.predict.side_effect = Exception("Model inference failed")

        with pytest.raises(PredictionException) as exc_info:
            service.predict("test_user")

        assert "Unexpected error during prediction" in str(exc_info.value)

    def test_predict_uses_latest_order(
        self, mock_model, mock_feature_store, mock_model_logger
    ):
        """Test that prediction uses the latest order (iloc[-1])"""
        service = PredictionService(mock_model_logger)
        service.model = mock_model
        service.feature_store = mock_feature_store

        service.predict("test_user")

        model_input = mock_model.predict.call_args[0][0]
        expected_values = np.array([[69.38, 11, 0, 0]])
        np.testing.assert_array_equal(model_input, expected_values)

    def test_feature_names_initialized(self, mock_model_logger):
        """Test that feature names are correctly initialized"""
        service = PredictionService(mock_model_logger)

        expected_features = [
            "prior_basket_value",
            "prior_item_count",
            "prior_regulars_count",
            "regulars_count",
        ]
        assert service.feature_names == expected_features

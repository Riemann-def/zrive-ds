# tests/module_7/test_prediction_service.py
import pytest
import numpy as np

from src.module_7.services.prediction_service import PredictionService
from src.module_7.exceptions.custom_exceptions import (
    UserNotFoundException,
    PredictionException,
)


class TestPredictionService:
    def test_predict_success(self, mock_model, mock_feature_store, mock_model_logger):
        """Test successful prediction"""
        # Arrange
        service = PredictionService(mock_model_logger)
        service.model = mock_model
        service.feature_store = mock_feature_store

        user_id = "test_user_123"
        expected_prediction = 75.42

        # Act
        result = service.predict(user_id)

        # Assert
        assert isinstance(result, float)
        assert result == expected_prediction
        mock_feature_store.get_features.assert_called_once_with(user_id)
        mock_model.predict.assert_called_once()

        # Verify model input shape is correct (1, 4)
        model_input = mock_model.predict.call_args[0][0]
        assert model_input.shape == (1, 4)

    def test_predict_user_not_found(
        self, mock_model, mock_feature_store, mock_model_logger
    ):
        """Test prediction when user doesn't exist"""
        # Arrange
        service = PredictionService(mock_model_logger)
        service.model = mock_model
        service.feature_store = mock_feature_store

        # Mock feature store to raise UserNotFoundException
        mock_feature_store.get_features.side_effect = UserNotFoundException(
            "User not found"
        )

        # Act & Assert
        with pytest.raises(UserNotFoundException):
            service.predict("nonexistent_user")

    def test_predict_model_error(
        self, mock_model, mock_feature_store, mock_model_logger
    ):
        """Test prediction when model fails"""
        # Arrange
        service = PredictionService(mock_model_logger)
        service.model = mock_model
        service.feature_store = mock_feature_store

        # Mock model to raise an exception
        mock_model.predict.side_effect = Exception("Model inference failed")

        # Act & Assert
        with pytest.raises(PredictionException) as exc_info:
            service.predict("test_user")

        assert "Unexpected error during prediction" in str(exc_info.value)

    def test_predict_uses_latest_order(
        self, mock_model, mock_feature_store, mock_model_logger
    ):
        """Test that prediction uses the latest order (iloc[-1])"""
        # Arrange
        service = PredictionService(mock_model_logger)
        service.model = mock_model
        service.feature_store = mock_feature_store

        # Act
        service.predict("test_user")

        # Assert - verify the model was called with the last row's values
        model_input = mock_model.predict.call_args[0][0]
        expected_values = np.array([[69.38, 11, 0, 0]])  # Last row values
        np.testing.assert_array_equal(model_input, expected_values)

    def test_feature_names_initialized(self, mock_model_logger):
        """Test that feature names are correctly initialized"""
        # Act
        service = PredictionService(mock_model_logger)

        # Assert
        expected_features = [
            "prior_basket_value",
            "prior_item_count",
            "prior_regulars_count",
            "regulars_count",
        ]
        assert service.feature_names == expected_features

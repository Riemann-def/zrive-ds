# tests/module_7/test_models.py
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.module_7.models.request import PredictRequest
from src.module_7.models.response import PredictResponse, StatusResponse


class TestModels:
    def test_predict_request_valid(self):
        """Test valid PredictRequest"""
        # Act
        request = PredictRequest(user_id="test_user_123")

        # Assert
        assert request.user_id == "test_user_123"

    def test_predict_request_missing_user_id(self):
        """Test PredictRequest with missing user_id"""
        # Act & Assert
        with pytest.raises(ValidationError):
            PredictRequest()

    def test_predict_response_valid(self):
        """Test valid PredictResponse"""
        # Arrange
        timestamp = datetime.now()

        # Act
        response = PredictResponse(
            user_id="test_user_123", predicted_price=75.42, timestamp=timestamp
        )

        # Assert
        assert response.user_id == "test_user_123"
        assert response.predicted_price == 75.42
        assert response.timestamp == timestamp

    def test_status_response_valid(self):
        """Test valid StatusResponse"""
        # Arrange
        timestamp = datetime.now()

        # Act
        response = StatusResponse(status="OK", timestamp=timestamp)

        # Assert
        assert response.status == "OK"
        assert response.timestamp == timestamp

# tests/module_7/test_endpoints.py
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from src.module_7.exceptions.custom_exceptions import (
    UserNotFoundException,
)


class TestEndpoints:
    def test_get_status_success(self):
        """Test GET /status endpoint"""
        from src.module_7.app import app

        client = TestClient(app)

        # Act
        response = client.get("/status")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert "timestamp" in data

    @patch("src.module_7.app.prediction_service")
    def test_predict_success(self, mock_prediction_service):
        """Test successful POST /predict"""
        from src.module_7.app import app

        client = TestClient(app)

        # Arrange
        mock_prediction_service.predict.return_value = 75.42

        # Act
        response = client.post("/predict", json={"user_id": "test_user_123"})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test_user_123"
        assert data["predicted_price"] == 75.42
        assert "timestamp" in data
        mock_prediction_service.predict.assert_called_once_with("test_user_123")

    def test_predict_missing_user_id(self):
        """Test POST /predict with missing user_id"""
        from src.module_7.app import app

        client = TestClient(app)

        # Act
        response = client.post("/predict", json={})

        # Assert
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_predict_empty_user_id(self):
        """Test POST /predict with empty user_id"""
        from src.module_7.app import app

        client = TestClient(app)

        # Act
        response = client.post("/predict", json={"user_id": ""})

        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Invalid user ID" in data["detail"]

    def test_predict_user_not_found(self):
        """Test POST /predict when user doesn't exist"""
        # Mock antes de importar
        with patch("services.prediction_service.PredictionService") as MockService:
            # Setup mock para lanzar excepción
            mock_instance = Mock()
            mock_instance.predict.side_effect = UserNotFoundException("User not found")
            MockService.return_value = mock_instance

            from src.module_7.app import app

            client = TestClient(app)

            # Act
            response = client.post("/predict", json={"user_id": "nonexistent_user"})

            # Assert
            assert response.status_code == 404
            data = response.json()
            assert "User not found" in data["detail"]

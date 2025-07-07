# tests/module_7/conftest.py
import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "module_7")
)


# Mock data fixtures
@pytest.fixture
def mock_features_data():
    """Mock feature data for testing"""
    return pd.DataFrame(
        {
            "prior_basket_value": [69.38, 94.61],
            "prior_item_count": [11, 15],
            "prior_regulars_count": [0, 0],
            "regulars_count": [0, 0],
        }
    )


@pytest.fixture
def mock_user_features():
    """Mock single user features (Series)"""
    return pd.Series(
        {
            "prior_basket_value": 69.38,
            "prior_item_count": 11,
            "prior_regulars_count": 0,
            "regulars_count": 0,
        }
    )


@pytest.fixture
def mock_model():
    """Mock model that returns predictable values"""
    model = Mock()
    model.predict.return_value = np.array([75.42])
    return model


@pytest.fixture
def mock_feature_store():
    """Mock feature store"""
    feature_store = Mock()
    mock_data = pd.DataFrame(
        {
            "prior_basket_value": [50.0, 69.38],
            "prior_item_count": [8, 11],
            "prior_regulars_count": [1, 0],
            "regulars_count": [2, 0],
        }
    )
    feature_store.get_features.return_value = mock_data
    return feature_store


@pytest.fixture
def mock_model_logger():
    """Mock model logger"""
    return Mock()


@pytest.fixture
def test_client():
    """FastAPI test client"""
    with patch("src.module_7.app.prediction_service"):
        from src.module_7.app import app

        return TestClient(app)

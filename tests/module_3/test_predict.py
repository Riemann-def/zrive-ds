import pandas as pd
import numpy as np
import pytest
from src.module_3.predict import preprocess_new_data, predict


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "user_id": [101, 102, 103, 104, 105],
            "product_type": ["A", "B", "C", "A", "B"],
            "vendor": ["X", "Y", "Z", "X", "Y"],
            "created_at": ["2020-10-05 16:46:19"] * 5,
        }
    )
    return df


@pytest.fixture
def mock_model():
    class MockModel:
        def predict_proba(self, X):
            # Returns probabilities proportional to index
            probs = np.array([i / 10 for i in range(len(X))])
            return np.vstack((1 - probs, probs)).T

    return MockModel()


def test_preprocess_new_data(sample_data):
    X = preprocess_new_data(sample_data)

    assert "order_id" not in X.columns
    assert "user_id" not in X.columns

    assert "product_type" in X.columns
    assert "vendor" in X.columns
    assert "created_at" in X.columns


def test_predict(mock_model, sample_data):
    X = preprocess_new_data(sample_data)

    predictions, probabilities = predict(mock_model, X, threshold=0.3)

    # With our mock model and threshold 0.3, indices 3 and 4 should be predicted as 1
    assert predictions[0] == 0
    assert predictions[1] == 0
    assert predictions[2] == 0
    assert predictions[3] == 1
    assert predictions[4] == 1

    # Probabilities should increase with index
    assert all(
        probabilities[i] < probabilities[i + 1] for i in range(len(probabilities) - 1)
    )

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.module_3.config import Config
from src.module_3.train import prepare_features, create_pipeline


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2, 3],
            "user_id": [101, 101, 102, 102, 103],
            "variant_id": [201, 202, 203, 204, 205],
            "product_type": ["A", "B", "A", "C", "B"],
            "vendor": ["X", "Y", "X", "Z", "Y"],
            "created_at": ["2020-10-05 16:46:19"] * 5,
            "order_date": ["2020-10-05 00:00:00"] * 5,
            "outcome": [0, 1, 0, 1, 0],
            "items_bought": [0, 5, 0, 6, 0],
        }
    )
    return df


def test_prepare_features(sample_data):
    X, y = prepare_features(sample_data)

    assert len(X) == len(sample_data)
    assert len(y) == len(sample_data)

    for col in Config.COLS_TO_DROP:
        assert col not in X.columns
    assert Config.TARGET_COL not in X.columns

    assert y.name == Config.TARGET_COL


def test_create_pipeline():
    pipeline = create_pipeline()

    assert isinstance(pipeline, Pipeline)

    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names
    assert "classifier" in step_names

    preprocessor = pipeline.named_steps["preprocessor"]
    preprocessor_step_names = [name for name, _ in preprocessor.steps]
    assert "date_transformer" in preprocessor_step_names
    assert "target_encoder" in preprocessor_step_names

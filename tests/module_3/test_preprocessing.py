import pandas as pd
from src.module_3.preprocessing import DateFeatureTransformer, TargetEncoder


def test_date_feature_transformer():
    data = pd.DataFrame({"created_at": ["2020-10-05 16:46:19", "2020-12-25 10:30:00"]})

    transformer = DateFeatureTransformer(date_column="created_at")
    result = transformer.fit_transform(data)

    assert "created_at" not in result.columns
    assert "year" in result.columns
    assert "month_num" in result.columns
    assert "is_weekend" in result.columns

    assert result.iloc[0]["year"] == 2020
    assert result.iloc[0]["month_num"] == 10
    assert result.iloc[1]["month_num"] == 12
    assert result.iloc[1]["is_weekend"] == 0  # December 25, 2020 was a Friday


def test_target_encoder():
    X = pd.DataFrame({"category": ["A", "A", "B", "B", "C"]})
    y = pd.Series([1, 1, 0, 0, 1])

    encoder = TargetEncoder(categorical_columns=["category"])
    encoder.fit(X, y)
    result = encoder.transform(X)

    assert "category" in result.columns

    assert result.iloc[0]["category"] > 0.5  # Category A
    assert result.iloc[2]["category"] < 0.5  # Category B

    X_test = pd.DataFrame({"category": ["D"]})
    result_test = encoder.transform(X_test)

    assert abs(result_test.iloc[0]["category"] - encoder.global_mean) < 1e-6

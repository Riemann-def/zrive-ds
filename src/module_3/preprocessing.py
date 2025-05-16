import re
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column="created_at"):
        self.date_column = date_column

    def process_date(self, input_str: str) -> dict:
        date_str = input_str.split(" ")[0]

        regex = re.compile(r"\d{4}-\d{2}-\d{2}")
        if not re.match(regex, date_str):
            return {}

        my_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        date_feats = {}

        date_feats["year"] = int(my_date.strftime("%Y"))
        date_feats["month_num"] = int(my_date.strftime("%m"))
        date_feats["dom"] = int(my_date.strftime("%d"))
        date_feats["doy"] = int(my_date.strftime("%j"))
        date_feats["woy"] = int(my_date.strftime("%W"))

        # Fixing day of week to start on Mon (1), end on Sun (7)
        dow = my_date.strftime("%w")
        if dow == "0":
            dow = 7
        date_feats["dow_num"] = int(dow)

        date_feats["is_weekend"] = 1 if int(dow) > 5 else 0

        return date_feats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        date_features = []
        for date_value in X_copy[self.date_column]:
            date_features.append(self.process_date(date_value))

        date_df = pd.DataFrame(date_features, index=X_copy.index)

        X_transformed = pd.concat([X_copy, date_df], axis=1)

        X_transformed = X_transformed.drop(columns=[self.date_column])

        return X_transformed


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.frequency_maps = {}

    def fit(self, X, y=None):
        for column in self.categorical_columns:
            if column in X.columns:
                frequencies = X[column].value_counts(normalize=True).to_dict()
                self.frequency_maps[column] = frequencies
        return self

    def transform(self, X):
        X_copy = X.copy()

        for column in self.categorical_columns:
            if column in X_copy.columns:
                default_value = (
                    min(self.frequency_maps[column].values())
                    if self.frequency_maps[column]
                    else 0
                )
                X_copy[column] = X_copy[column].map(
                    lambda x: self.frequency_maps[column].get(x, default_value)
                )

        return X_copy


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns, target_column="outcome", smoothing=1.0):
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.smoothing = smoothing
        self.target_means = {}
        self.global_mean = 0

    def fit(self, X, y=None):
        self.global_mean = y.mean()

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        for column in self.categorical_columns:
            if column in X.columns:
                temp_df = pd.DataFrame({"category": X[column], "target": y})
                agg_df = temp_df.groupby("category").agg({"target": ["count", "mean"]})
                agg_df.columns = ["count", "mean"]
                agg_df["smooth_mean"] = (
                    agg_df["count"] * agg_df["mean"] + self.smoothing * self.global_mean
                ) / (agg_df["count"] + self.smoothing)

                self.target_means[column] = agg_df["smooth_mean"].to_dict()

        return self

    def transform(self, X):
        X_copy = X.copy()

        for column in self.categorical_columns:
            if column in X_copy.columns:
                X_copy[column] = X_copy[column].map(
                    lambda x: self.target_means[column].get(x, self.global_mean)
                )

        return X_copy

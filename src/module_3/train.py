import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from .config import Config
from .preprocessing import DateFeatureTransformer, TargetEncoder


logger = Config.setLogger("train_logger")


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    return df


def filter_for_minimum_items(df: pd.DataFrame, min_items: int = 5) -> pd.DataFrame:
    items_bought_per_order = (
        df[df["outcome"] == 1]
        .groupby("order_id")
        .size()
        .reset_index(name="items_bought")
    )
    df_with_count = df.merge(items_bought_per_order, on="order_id", how="left")
    df_with_count["items_bought"].fillna(0, inplace=True)

    filtered_df = df_with_count[df_with_count["items_bought"] > (min_items - 1)]
    logger.info(f"Filtered data: {len(filtered_df)} rows")
    return filtered_df


def prepare_features(df: pd.DataFrame) -> tuple:
    feature_cols = df.columns[
        ~df.columns.isin(Config.COLS_TO_DROP + [Config.TARGET_COL])
    ]
    X = df[feature_cols]
    y = df[Config.TARGET_COL]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    gss1 = GroupShuffleSplit(
        n_splits=1, train_size=Config.TRAIN_SIZE, random_state=Config.RANDOM_STATE
    )
    train_idx, temp_idx = next(gss1.split(X, y, groups=X[Config.ORDER_ID_COL]))

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()

    X_temp = X.iloc[temp_idx].copy()
    y_temp = y.iloc[temp_idx].copy()
    groups_temp = X_temp[Config.ORDER_ID_COL].copy()

    remaining_proportion = 1 - Config.TRAIN_SIZE
    val_proportion_of_remaining = Config.VAL_SIZE / remaining_proportion

    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_proportion_of_remaining,
        random_state=Config.RANDOM_STATE,
    )
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))

    X_val = X_temp.iloc[val_idx].copy()
    y_val = y_temp.iloc[val_idx].copy()

    X_test = X_temp.iloc[test_idx].copy()
    y_test = y_temp.iloc[test_idx].copy()

    X_train.drop(columns=[Config.ORDER_ID_COL], inplace=True)
    X_val.drop(columns=[Config.ORDER_ID_COL], inplace=True)
    X_test.drop(columns=[Config.ORDER_ID_COL], inplace=True)

    logger.info(
        f"Train: {len(X_train)} samples, "
        f"Val: {len(X_val)} samples, "
        f"Test: {len(X_test)} samples"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_pipeline() -> Pipeline:
    preprocessing_pipeline = Pipeline(
        [
            ("date_transformer", DateFeatureTransformer(date_column=Config.DATE_COL)),
            (
                "target_encoder",
                TargetEncoder(categorical_columns=Config.CATEGORICAL_COLS),
            ),
        ]
    )

    model_pipeline = Pipeline(
        [
            ("preprocessor", preprocessing_pipeline),
            (
                "classifier",
                LogisticRegression(
                    penalty=None,
                    class_weight=Config.CLASS_WEIGHT,
                    max_iter=1000,
                    random_state=Config.RANDOM_STATE,
                ),
            ),
        ]
    )

    return model_pipeline


def find_optimal_threshold(
    model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
) -> float:
    y_val_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)

    f_beta_scores = ((1 + Config.F_BETA**2) * precision * recall) / (
        (Config.F_BETA**2 * precision) + recall + 1e-10
    )
    best_idx = np.argmax(f_beta_scores)

    if best_idx < len(thresholds):
        return thresholds[best_idx]
    return 0.5


def train_and_save_model():
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    try: 
        # Load and prepare data
        df = load_data(Config.MODULE_DIR + Config.LOCAL_DATA_PATH)
        filtered_df = filter_for_minimum_items(df)
        X, y = prepare_features(filtered_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Train model
        logger.info("Training model...")
        pipeline = create_pipeline()
        pipeline.fit(X_train[:1000], y_train[:1000])

        # Find optimal threshold
        logger.info("Finding optimal threshold...")
        threshold = find_optimal_threshold(pipeline, X_val, y_val)
        logger.info(f"Optimal threshold: {threshold:.4f}")

        # Save model and threshold
        logger.info(f"Saving model to {Config.MODEL_FILE}")
        joblib.dump(pipeline, Config.MODEL_FILE)

        with open(Config.THRESHOLD_FILE, "w") as f:
            f.write(str(threshold))

        logger.info("Model and threshold saved successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    train_and_save_model()

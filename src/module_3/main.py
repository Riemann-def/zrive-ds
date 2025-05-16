import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    fbeta_score,
)

from .config import Config
from .preprocessing import DateFeatureTransformer, TargetEncoder, FrequencyEncoder


logger = Config.setLogger("train_logger")


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        Config.TARGET_COL,
        Config.ORDER_ID_COL,
    ] + Config.CATEGORICAL_COLS
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info(f"Data loaded successfully: {len(df)} rows")
    return df


def filter_for_minimum_items(df: pd.DataFrame, min_items: int = 5) -> pd.DataFrame:
    items_bought_per_order = (
        df[df["outcome"] == 1]
        .groupby("order_id")
        .size()
        .reset_index(name="items_bought")
    )
    df_with_count = df.merge(items_bought_per_order, on="order_id", how="left")
    df_with_count["items_bought"] = df_with_count["items_bought"].fillna(0)

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


def create_candidate_pipelines():
    frequency_encoding_pipeline = Pipeline(
        [
            ("date_transformer", DateFeatureTransformer(date_column=Config.DATE_COL)),
            (
                "frequency_encoder",
                FrequencyEncoder(categorical_columns=Config.CATEGORICAL_COLS),
            ),
        ]
    )

    target_encoding_pipeline = Pipeline(
        [
            ("date_transformer", DateFeatureTransformer(date_column=Config.DATE_COL)),
            (
                "target_encoder",
                TargetEncoder(categorical_columns=Config.CATEGORICAL_COLS),
            ),
        ]
    )

    pipeline1 = Pipeline(
        [
            ("preprocessor", frequency_encoding_pipeline),
            (
                "classifier",
                LogisticRegression(
                    penalty=None,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=Config.RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline2 = Pipeline(
        [
            ("preprocessor", target_encoding_pipeline),
            (
                "classifier",
                LogisticRegression(
                    penalty=None,
                    class_weight={0: 1, 1: 5},
                    max_iter=1000,
                    random_state=Config.RANDOM_STATE,
                ),
            ),
        ]
    )

    return [
        ("frequency_encoding_balanced", pipeline1),
        ("target_encoding_custom_weight", pipeline2),
    ]


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
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5

    return (
        best_threshold,
        precision[best_idx],
        recall[best_idx],
        f_beta_scores[best_idx],
    )


def compare_models(pipelines, X_train, y_train, X_val, y_val):
    results = []

    for name, pipeline in pipelines:
        logger.info(f"Evaluating pipeline: {name}")

        pipeline.fit(X_train, y_train)
        threshold, precision, recall, f_beta = find_optimal_threshold(
            pipeline, X_val, y_val
        )

        results.append(
            {
                "name": name,
                "pipeline": pipeline,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f_beta": f_beta,
            }
        )

        logger.info(f"  Threshold: {threshold:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F{Config.F_BETA} Score: {f_beta:.4f}")

    results.sort(key=lambda x: x["f_beta"], reverse=True)

    best_model = results[0]
    logger.info(
        f"Best model: {best_model['name']} with F{Config.F_BETA} "
        f"score: {best_model['f_beta']:.4f}"
    )

    return best_model, results


def train_final_model(best_pipeline, X_train, y_train, X_val, y_val):
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    logger.info(f"Training final model on combined data: {len(X_combined)} samples")

    best_pipeline.fit(X_combined, y_combined)

    return best_pipeline


def evaluate_on_test(model, threshold, X_test, y_test):
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f_beta = fbeta_score(y_test, y_test_pred, beta=Config.F_BETA)

    logger.info("Test set evaluation:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F{Config.F_BETA} Score: {f_beta:.4f}")

    return {"precision": precision, "recall": recall, f"f{Config.F_BETA}_score": f_beta}


def save_model(model, threshold, test_metrics, training_history):
    paths = Config.get_model_version_path()
    model_path, threshold_path, version, metadata_path = paths

    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    with open(threshold_path, "w") as f:
        f.write(str(threshold))

    metadata = {
        "performance": {
            "test": test_metrics,
            "validation": {
                "precision": float(training_history["precision"]),
                "recall": float(training_history["recall"]),
                f"f{Config.F_BETA}_score": float(training_history["f_beta"]),
            },
            "threshold": float(threshold),
        },
        "model_info": {
            "name": training_history["name"],
            "pipeline_steps": str([step[0] for step in model.steps]),
        },
        "parameters": {
            "random_state": Config.RANDOM_STATE,
            "class_weight": str(Config.CLASS_WEIGHT),
            "f_beta": Config.F_BETA,
        },
        "comparison_results": [
            {
                "name": result["name"],
                "f_beta": float(result["f_beta"]),
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "threshold": float(result["threshold"]),
            }
            for result in training_history["all_results"]
        ],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    Config.register_model(version, metadata)

    logger.info(f"Model version {version} saved and set as active")

    return version


def select_and_train_best_model():
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    try:
        loaded_df = load_data(Config.MODULE_DIR + Config.LOCAL_DATA_PATH)
        validated_df = validate_data(loaded_df)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    filtered_df = filter_for_minimum_items(validated_df)

    X, y = prepare_features(filtered_df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    pipelines = create_candidate_pipelines()

    best_model, all_results = compare_models(pipelines, X_train, y_train, X_val, y_val)

    final_model = train_final_model(
        best_model["pipeline"], X_train, y_train, X_val, y_val
    )

    test_metrics = evaluate_on_test(
        final_model, best_model["threshold"], X_test, y_test
    )

    training_history = {
        "name": best_model["name"],
        "precision": best_model["precision"],
        "recall": best_model["recall"],
        "f_beta": best_model["f_beta"],
        "all_results": all_results,
    }

    version = save_model(
        final_model, best_model["threshold"], test_metrics, training_history
    )

    return version


if __name__ == "__main__":
    try:
        version = select_and_train_best_model()
        logger.info(f"Model training completed successfully! Model version: {version}")
    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}")
        raise

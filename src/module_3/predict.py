import joblib
import pandas as pd
import numpy as np
from .config import Config

logger = Config.setLogger("predict_logger")


def load_model():
    logger.info(f"Loading model from {Config.MODEL_FILE}")
    model = joblib.load(Config.MODEL_FILE)

    with open(Config.THRESHOLD_FILE, "r") as f:
        threshold = float(f.read().strip())

    return model, threshold


def preprocess_new_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns and ensure order_id is removed for prediction
    cols_to_drop = Config.COLS_TO_DROP.copy()
    if Config.ORDER_ID_COL in df.columns:
        if Config.ORDER_ID_COL not in cols_to_drop:
            cols_to_drop.append(Config.ORDER_ID_COL)

    # Ensure the target column is not in the input
    if Config.TARGET_COL in df.columns:
        if Config.TARGET_COL not in cols_to_drop:
            cols_to_drop.append(Config.TARGET_COL)

    # Return only needed columns
    available_cols = [col for col in df.columns if col not in cols_to_drop]
    return df[available_cols]


def predict(model, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    # Get probability predictions
    probabilities = model.predict_proba(df)[:, 1]

    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities


def predict_from_file(filepath: str) -> pd.DataFrame:
    # Load model and threshold
    model, threshold = load_model()

    # Load and preprocess data
    df = pd.read_csv(filepath)
    X = preprocess_new_data(df)

    # Get predictions
    predictions, probabilities = predict(model, X, threshold)

    # Add predictions to the original dataframe
    results_df = df.copy()
    results_df["prediction"] = predictions
    results_df["probability"] = probabilities

    logger.info(f"Made predictions for {len(df)} samples")
    logger.info(
        f"Predicted purchases: {predictions.sum()} ({predictions.mean()*100:.2f}%)"
    )

    return results_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = Config.LOCAL_DATA_PATH

    results = predict_from_file(filepath)

    # If needed, save results
    output_path = filepath.replace(".csv", "_predictions.csv")
    results.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

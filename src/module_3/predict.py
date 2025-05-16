import pandas as pd
import joblib
from .config import Config

logger = Config.setLogger("predict_logger")


def load_model():
    try:
        model_path, threshold_path = Config.get_active_model_paths()

        logger.info(f"Loading active model from {model_path}")
        model = joblib.load(model_path)

        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())

        return model, threshold

    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise


def preprocess_data(df):
    cols_to_drop = Config.COLS_TO_DROP.copy()

    if Config.ORDER_ID_COL in df.columns and Config.ORDER_ID_COL not in cols_to_drop:
        cols_to_drop.append(Config.ORDER_ID_COL)

    if Config.TARGET_COL in df.columns and Config.TARGET_COL not in cols_to_drop:
        cols_to_drop.append(Config.TARGET_COL)

    drop_cols = [col for col in cols_to_drop if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def predict(data:pd.DataFrame, threshold:float=None)-> tuple:
    model, default_threshold = load_model()

    if threshold is None:
        threshold = default_threshold

    X = preprocess_data(data)

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities


def predict_from_file(filepath: str, output_path: str = None
                      ) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    predictions, probabilities = predict(df)

    results = df.copy()
    results["purchase_probability"] = probabilities
    results["predicted_purchase"] = predictions

    logger.info(f"Made predictions for {len(df)} rows")
    logger.info(
        f"Predicted purchases: {predictions.sum()} ({predictions.mean()*100:.2f}%)"
    )

    if output_path:
        results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = input_file.replace(".csv", "_predictions.csv")

        try:
            predict_from_file(input_file, output_file)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            sys.exit(1)
    else:
        logger.error("No input file specified")
        print("Usage: python -m src.module_3.predict path/to/input.csv")
        sys.exit(1)

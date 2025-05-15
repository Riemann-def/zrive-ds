import os
import logging


class Config:
    MODULE_DIR = "src/module_3/"
    DATA_DIR = "data/"
    LOCAL_DATA_PATH = os.path.join(DATA_DIR, "feature_frame.csv")
    MODELS_DIR = "models/"
    MODEL_FILE = os.path.join(MODELS_DIR, "classification_pipeline.pkl")
    THRESHOLD_FILE = os.path.join(MODELS_DIR, "optimal_threshold.txt")

    RANDOM_STATE = 55
    ORDER_ID_COL = "order_id"
    TARGET_COL = "outcome"
    CATEGORICAL_COLS = ["product_type", "vendor"]
    DATE_COL = "created_at"
    COLS_TO_DROP = ["variant_id", "user_id", "order_date", "items_bought"]

    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.2
    F_BETA = 0.3
    CLASS_WEIGHT = {0: 1, 1: 5}

    @staticmethod
    def setLogger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

import os
import logging
from datetime import datetime
import json
from pathlib import Path


class Config:
    MODULE_DIR = "src/module_3/"
    DATA_DIR = "data/"
    LOCAL_DATA_PATH = os.path.join(DATA_DIR, "feature_frame.csv")
    MODELS_DIR = "models/"

    MODEL_REGISTRY_FILE = os.path.join(MODELS_DIR, "model_registry.json")

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
    def get_model_version_path(version: str = None) -> tuple:
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"

        version_dir = os.path.join(Config.MODELS_DIR, version)
        os.makedirs(version_dir, exist_ok=True)

        model_path = os.path.join(version_dir, "model.pkl")
        threshold_path = os.path.join(version_dir, "threshold.txt")
        metadata_path = os.path.join(version_dir, "metadata.json")

        return model_path, threshold_path, version, metadata_path

    @staticmethod
    def register_model(version: str, metadata: dict) -> None:
        registry_path = Path(Config.MODEL_REGISTRY_FILE)
        registry = {}

        os.makedirs(Config.MODELS_DIR, exist_ok=True)

        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)

        registry[version] = {
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            **metadata,
        }

        for other_version in registry:
            if other_version != version:
                registry[other_version]["is_active"] = False

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        model_path, threshold_path, _, _ = Config.get_model_version_path(version)

        active_model_path = os.path.join(Config.MODELS_DIR, "active_model.pkl")
        active_threshold_path = os.path.join(Config.MODELS_DIR, "active_threshold.txt")

        if os.path.exists(active_model_path):
            os.remove(active_model_path)
        if os.path.exists(active_threshold_path):
            os.remove(active_threshold_path)

        os.symlink(os.path.relpath(model_path, Config.MODELS_DIR), active_model_path)
        os.symlink(
            os.path.relpath(threshold_path, Config.MODELS_DIR), active_threshold_path
        )

    @staticmethod
    def get_active_model_paths():
        model_path = os.path.join(Config.MODELS_DIR, "active_model.pkl")
        threshold_path = os.path.join(Config.MODELS_DIR, "active_threshold.txt")

        if not os.path.exists(model_path) or not os.path.exists(threshold_path):
            registry_path = Path(Config.MODEL_REGISTRY_FILE)

            if not registry_path.exists():
                raise FileNotFoundError("No models have been registered yet.")

            with open(registry_path, "r") as f:
                registry = json.load(f)

            active_versions = [
                v for v, data in registry.items() if data.get("is_active", False)
            ]

            if not active_versions:
                raise FileNotFoundError("No active model found in registry.")

            return Config.get_model_version_path(active_versions[0])[:2]

        return model_path, threshold_path

    @staticmethod
    def setLogger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:  # Only add handler if it doesn't already have one
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

# logger.py
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from config import Config


class ServiceLogger:
    def __init__(self, log_file=None):
        if log_file is None:
            log_file = Config.SERVICE_LOG_FILE

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("SERVICE METRICS LOG\n")
                f.write("=" * 50 + "\n")

    def log_request(
        self, method: str, endpoint: str, latency: float, status_code: str
    ) -> None:
        """HTTP request logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"{timestamp} | REQUEST | {method} {endpoint} | "
            f"latency={latency:.3f}s | status={status_code}\n"
        )

        with open(self.log_file, "a") as f:
            f.write(log_entry)

    def log_error(self, error_message: str) -> None:
        """Service error logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | ERROR | {error_message}\n"

        with open(self.log_file, "a") as f:
            f.write(log_entry)

    def log_info(self, info_message: str) -> None:
        """General information logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | INFO | {info_message}\n"

        with open(self.log_file, "a") as f:
            f.write(log_entry)


class ModelLogger:
    def __init__(self, log_file=None):
        if log_file is None:
            log_file = Config.MODEL_LOG_FILE

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("MODEL METRICS LOG\n")
                f.write("=" * 50 + "\n")
                f.write(
                    "Format: TIMESTAMP | PREDICTION | user_id | prediction | "
                    "latency | total_orders | input_features | feature_stats\n"
                )
                f.write("=" * 50 + "\n")

    def log_complete_prediction(
        self,
        user_id: str,
        prediction: float,
        latency: float,
        raw_features: pd.DataFrame,
        selected_features: pd.Series,
        feature_names: list,
        input_features: np.ndarray,
    ) -> None:
        """Log everything about a prediction in one comprehensive line"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Input features como pares nombre=valor
        features_str = " | ".join(
            [
                f"{name}={value:.3f}"
                for name, value in zip(feature_names, input_features.flatten())
            ]
        )

        # Stats básicas de las features seleccionadas
        stats = {
            "mean": selected_features.mean(),
            "std": selected_features.std(),
            "min": selected_features.min(),
            "max": selected_features.max(),
        }
        stats_str = ",".join([f"{k}={v:.2f}" for k, v in stats.items()])

        # Log completo en una línea
        log_entry = (
            f"{timestamp} | PREDICTION | "
            f"user_id={user_id} | "
            f"prediction={prediction:.2f} | "
            f"latency={latency:.3f}s | "
            f"total_orders={len(raw_features)} | "
            f"input_features=({features_str}) | "
            f"stats=({stats_str})\n"
        )

        with open(self.log_file, "a") as f:
            f.write(log_entry)

    def log_error(self, user_id: str, error_type: str, error_message: str):
        """Log model-related errors"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"{timestamp} | MODEL_ERROR | user_id={user_id} | "
            f"type={error_type} | error={error_message}\n"
        )

        with open(self.log_file, "a") as f:
            f.write(log_entry)

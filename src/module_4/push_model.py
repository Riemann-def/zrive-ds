from pathlib import Path
from typing import Any, Dict, Union
import glob
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import fbeta_score

from src.module_4.config import THRESHOLD, MODEL_DIR, TODAY_STR


class PushModel:
    DEFAULT_PARAMS: Dict[str, Any] = {
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.005,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    }

    def __init__(
        self, params: Dict[str, Any] | None = None, *, threshold: float = THRESHOLD
    ):
        self.params: Dict[str, Any] = {**self.DEFAULT_PARAMS, **(params or {})}
        self.threshold: float = threshold
        self._model: LGBMClassifier = self._build_model()

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> "PushModel":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._model.predict_proba(X)[:, 1], index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: str | Path = None) -> Path:
        if path is None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = MODEL_DIR / f"push_model_{TODAY_STR}.pkl"
        else:
            path = Path(path)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PushModel":
        return joblib.load(path)

    @classmethod
    def latest_model_path(cls) -> Path:
        model_files = glob.glob(str(MODEL_DIR / "push_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No model files found")
        return Path(max(model_files))

    def fbeta_score(self, y_true, y_pred, beta=0.3):
        return fbeta_score(y_true, y_pred, beta=beta)

    def _build_model(self) -> LGBMClassifier:
        return LGBMClassifier(**self.params)

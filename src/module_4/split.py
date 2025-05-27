import pandas as pd
from src.module_4.config import TRAIN_SIZE, DATE_COL, ORDER_ID_COL, TARGET_COL

from typing import Tuple


def temporal_train_test_split(
    df: pd.DataFrame,
    train_size: float = TRAIN_SIZE,
    order_id_col: str = ORDER_ID_COL,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df_ = df.copy()
    df_[date_col] = pd.to_datetime(df_[date_col])

    order_dates = df_.groupby(order_id_col)[date_col].min().sort_values().reset_index()

    n_orders = len(order_dates)
    n_train_orders = int(n_orders * train_size)

    train_orders = set(order_dates.iloc[:n_train_orders][order_id_col])
    test_orders = set(order_dates.iloc[n_train_orders:][order_id_col])

    train_mask = df_[order_id_col].isin(train_orders)
    test_mask = df_[order_id_col].isin(test_orders)

    X = df_.drop(columns=[target_col])
    y = df_[target_col]

    X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
    y_train, y_test = y[train_mask].copy(), y[test_mask].copy()

    return X_train, X_test, y_train, y_test

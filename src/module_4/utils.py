import pandas as pd
import json
from pathlib import Path
from src.module_4.config import NUM_COLS, CAT_COLS, TARGET_COL, MIN_PURCHASES


def validate_event(event: dict, param_name: str) -> dict:
    if not isinstance(event, dict):
        raise ValueError("Event must be a dictionary.")

    if not isinstance(param_name, str):
        raise ValueError(f"Parameter {param_name} name must be a string.")

    params_str = event.get(param_name, None)
    if not params_str:
        raise ValueError(f"{param_name} json is required.")

    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format for {param_name}.")

    return params


def get_users_df(users: dict) -> pd.DataFrame:
    if not isinstance(users, dict):
        raise ValueError("Users must be a dictionary.")

    return pd.DataFrame.from_dict(users, orient="index")


def load_feature_frame(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        validate_df(df)
        filtered_df = filter_required_product_purchased(df, MIN_PURCHASES)
        return filtered_df

    except FileNotFoundError:
        raise FileNotFoundError(f"Feature file not found at {path}")

    except ValueError as e:
        raise ValueError(f"Error loading feature frame: {e}")

    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while loading the feature frame: {e}"
        )


def validate_df(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("DataFrame is empty.")

    if not all(col in df.columns for col in NUM_COLS + CAT_COLS):
        raise ValueError("DataFrame does not contain all required columns.")

    if df.isnull().values.all():
        raise ValueError("DataFrame contains only null values.")


def filter_required_product_purchased(
    df: pd.DataFrame, min_purchases: int
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty.")

    items_bought_per_order = (
        df[df[TARGET_COL] == 1]
        .groupby("order_id")
        .size()
        .reset_index(name="items_bought")
    )
    df_with_count = df.merge(items_bought_per_order, on="order_id", how="left")
    df_with_count["items_bought"].fillna(0, inplace=True)

    sales_df = df_with_count[df_with_count["items_bought"] >= min_purchases]

    return sales_df.drop(columns=["items_bought"])

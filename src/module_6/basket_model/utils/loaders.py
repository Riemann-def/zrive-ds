import os
import pandas as pd
from pathlib import Path
from config import Config


def load_orders() -> pd.DataFrame:
    orders_path = Config.DATA_DIR / "orders.parquet"
    if not orders_path.exists():
        raise FileNotFoundError(f"Orders file not found at {orders_path}")
    
    orders = pd.read_parquet(orders_path)
    orders = orders.sort_values(by=["user_id", "created_at"])
    orders["item_count"] = orders.apply(lambda x: len(x.item_ids), axis=1)
    orders["user_order_seq"] = (
        orders.groupby(["user_id"])["created_at"].rank().astype(int)
    )
    return orders

def load_regulars() -> pd.DataFrame:
    regulars_path = Config.DATA_DIR / "regulars.parquet"
    if not regulars_path.exists():
        raise FileNotFoundError(f"Regulars file not found at {regulars_path}")
    return pd.read_parquet(regulars_path)

def get_mean_item_price() -> float:
    inventory_path = Config.DATA_DIR / "inventory.parquet"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory file not found at {inventory_path}")
    inventory = pd.read_parquet(inventory_path)
    return inventory.our_price.mean()
import pandas as pd
from config import Config
from constants import DataFiles, Columns, CalculatedColumns


def load_orders() -> pd.DataFrame:
    try:
        orders_path = Config.DATA_DIR / DataFiles.ORDERS
        if not orders_path.exists():
            raise FileNotFoundError(f"Orders file not found at {orders_path}")

        orders = pd.read_parquet(orders_path)
        orders = orders.sort_values(
            by=[Columns.Orders.USER_ID, Columns.Orders.CREATED_AT]
        )
        orders[CalculatedColumns.ITEM_COUNT] = orders.apply(
            lambda x: len(x[Columns.Orders.ITEM_IDS]), axis=1
        )
        orders[CalculatedColumns.USER_ORDER_SEQ] = (
            orders.groupby([Columns.Orders.USER_ID])[Columns.Orders.CREATED_AT]
            .rank()
            .astype(int)
        )
        print(f"user_id: {orders[Columns.Orders.USER_ID].iloc[400]}")
        return orders

    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"An error occurred while loading orders: {str(e)}")


def load_regulars() -> pd.DataFrame:
    try:
        regulars_path = Config.DATA_DIR / DataFiles.REGULARS
        if not regulars_path.exists():
            raise FileNotFoundError(f"Regulars file not found at {regulars_path}")
        return pd.read_parquet(regulars_path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"An error occurred while loading regulars: {str(e)}")


def get_mean_item_price() -> float:
    try:
        inventory_path = Config.DATA_DIR / DataFiles.INVENTORY
        if not inventory_path.exists():
            raise FileNotFoundError(f"Inventory file not found at {inventory_path}")
        inventory = pd.read_parquet(inventory_path)
        return inventory[Columns.Inventory.PRICE].mean()
    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"An error occurred while loading inventory: {str(e)}")

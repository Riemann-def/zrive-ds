import numpy as np
import pandas as pd
from constants import Columns, CalculatedColumns


def count_regulars_in_order(order: pd.DataFrame, user_regulars: pd.DataFrame) -> int:
    try:
        return len(
            set(order[Columns.Orders.ITEM_IDS]).intersection(
                set(user_regulars[Columns.Regulars.VARIANT_ID].values)
            )
        )
    except Exception as e:
        raise Exception(f"Error counting regulars in order: {str(e)}")


def count_regulars_in_orders(
    orders: pd.DataFrame, regulars: pd.DataFrame
) -> np.ndarray:
    try:
        counts = []
        for _, order in orders.iterrows():
            user_regulars = regulars.loc[
                lambda x: (x[Columns.Regulars.USER_ID] == order[Columns.Orders.USER_ID])
            ]
            counts += [count_regulars_in_order(order, user_regulars)]
        return np.array(counts)
    except Exception as e:
        raise Exception(f"Error counting regulars in orders: {str(e)}")


def compute_basket_value(orders: pd.DataFrame, mean_item_price: float) -> pd.Series:
    try:
        return orders[CalculatedColumns.ITEM_COUNT] * mean_item_price
    except Exception as e:
        raise Exception(f"Error computing basket value: {str(e)}")


def enrich_orders(
    orders: pd.DataFrame, regulars: pd.DataFrame, mean_item_price: float
) -> pd.DataFrame:
    try:
        enriched_orders = orders.copy()
        enriched_orders[CalculatedColumns.REGULARS_COUNT] = count_regulars_in_orders(
            enriched_orders, regulars
        )
        enriched_orders[CalculatedColumns.BASKET_VALUE] = compute_basket_value(
            enriched_orders, mean_item_price
        )
        return enriched_orders

    except Exception as e:
        raise Exception(f"Error enriching orders: {str(e)}")


def build_prior_orders(enriched_orders: pd.DataFrame) -> pd.DataFrame:
    try:
        prior_orders = enriched_orders.copy()
        prior_orders[CalculatedColumns.USER_ORDER_SEQ_PLUS_1] = (
            prior_orders[CalculatedColumns.USER_ORDER_SEQ] + 1
        )
        prior_orders[CalculatedColumns.PRIOR_BASKET_VALUE] = prior_orders[
            CalculatedColumns.BASKET_VALUE
        ]
        prior_orders[CalculatedColumns.PRIOR_ITEM_COUNT] = prior_orders[
            CalculatedColumns.ITEM_COUNT
        ]
        prior_orders[CalculatedColumns.PRIOR_REGULARS_COUNT] = prior_orders[
            CalculatedColumns.REGULARS_COUNT
        ]
        return prior_orders.loc[
            :,
            [
                Columns.Orders.USER_ID,
                CalculatedColumns.USER_ORDER_SEQ_PLUS_1,
                CalculatedColumns.PRIOR_ITEM_COUNT,
                CalculatedColumns.PRIOR_REGULARS_COUNT,
                CalculatedColumns.PRIOR_BASKET_VALUE,
            ],
        ]
    except Exception as e:
        raise Exception(f"Error building prior orders: {str(e)}")


def build_feature_frame(
    orders: pd.DataFrame, regulars: pd.DataFrame, mean_item_price: float
) -> pd.DataFrame:
    try:
        enriched_orders = enrich_orders(orders, regulars, mean_item_price)
        prior_orders = build_prior_orders(enriched_orders)
        return pd.merge(
            enriched_orders.loc[
                :,
                [
                    Columns.Orders.USER_ID,
                    Columns.Orders.CREATED_AT,
                    CalculatedColumns.USER_ORDER_SEQ,
                    CalculatedColumns.BASKET_VALUE,
                    CalculatedColumns.REGULARS_COUNT,
                ],
            ],
            prior_orders,
            how="inner",
            left_on=(Columns.Orders.USER_ID, CalculatedColumns.USER_ORDER_SEQ),
            right_on=(Columns.Orders.USER_ID, CalculatedColumns.USER_ORDER_SEQ_PLUS_1),
        ).drop(
            [CalculatedColumns.USER_ORDER_SEQ, CalculatedColumns.USER_ORDER_SEQ_PLUS_1],
            axis=1,
        )
    except Exception as e:
        raise Exception(f"Error building feature frame: {str(e)}")

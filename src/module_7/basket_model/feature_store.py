import pandas as pd

from basket_model.utils import features
from basket_model.utils import loaders
from exceptions.custom_exceptions import UserNotFoundException
from constants import FeatureColumns, Columns


class FeatureStore:
    def __init__(self):
        orders = loaders.load_orders()
        regulars = loaders.load_regulars()
        mean_item_price = loaders.get_mean_item_price()
        self.feature_store = (
            features.build_feature_frame(orders, regulars, mean_item_price)
            .set_index(Columns.Orders.USER_ID)
            .loc[:, FeatureColumns.get_predictive_features()]
        )

    def get_features(self, user_id: str) -> pd.Series:
        try:
            features = self.feature_store.loc[user_id]
        except KeyError:
            raise UserNotFoundException(
                f"User ID '{user_id}' not found in feature store."
            )
        except Exception as e:
            raise UserNotFoundException(
                f"Error retrieving features for user {user_id}: {str(e)}"
            )
        return features

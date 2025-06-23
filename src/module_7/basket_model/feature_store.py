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
        print(
            f"Feature store initialized with {len(self.feature_store)} users."
        )
        print(
            f"Feature store columns: {self.feature_store.columns.tolist()}"
        )
        user_id = (
            "0cad47d6c469dc763adb1d206463099d94102e680eb2815de5d0ae055bdd72e2f2df884c327bfffbbefed72720bb6ec523e244912bf26aa78f5fe7d7ab893bc7"
        )
        print(
            f"loc user_id: {self.feature_store.loc[user_id]}"
        )
        print(
            f"Shape user_id_loc: {self.feature_store.loc[user_id].shape}"
        )

    def get_features(self, user_id: str) -> pd.Series:
        try:
            features = self.feature_store.loc[user_id]
        except Exception as e:
            raise UserNotFoundException(
                f"User not found in feature store: {str(e)}"
            )
        return features

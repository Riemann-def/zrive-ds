# constants.py

class DataFiles:
    """File names for data and model files"""
    ORDERS = "orders.parquet"
    REGULARS = "regulars.parquet"
    INVENTORY = "inventory.parquet"
    MODEL = "model.joblib"


class Columns:
    
    class Orders:
        ITEM_IDS = "ordered_items"
        USER_ID = "user_id"
        CREATED_AT = "created_at"
    
    class Regulars:
        USER_ID = "user_id"
        VARIANT_ID = "variant_id"
    
    class Inventory:
        VARIANT_ID = "variant_id"
        PRICE = "price"
        COMPARE_AT_PRICE = "compare_at_price"
        VENDOR = "vendor"
        PRODUCT_TYPE = "product_type"
        TAGS = "tags"


class CalculatedColumns:
    """Fields calculated during data processing"""
    ITEM_COUNT = "item_count"
    USER_ORDER_SEQ = "user_order_seq"
    USER_ORDER_SEQ_PLUS_1 = "user_order_seq_plus_1"
    BASKET_VALUE = "basket_value"
    REGULARS_COUNT = "regulars_count"
    
    PRIOR_BASKET_VALUE = "prior_basket_value"
    PRIOR_ITEM_COUNT = "prior_item_count"
    PRIOR_REGULARS_COUNT = "prior_regulars_count"


class FeatureColumns:
    """Final feature columns used by the model"""
    PRIOR_BASKET_VALUE = "prior_basket_value"
    PRIOR_ITEM_COUNT = "prior_item_count"
    PRIOR_REGULARS_COUNT = "prior_regulars_count"
    REGULARS_COUNT = "regulars_count"
    
    @classmethod
    def get_predictive_features(cls) -> list:
        return [
            cls.PRIOR_BASKET_VALUE,
            cls.PRIOR_ITEM_COUNT,
            cls.PRIOR_REGULARS_COUNT,
            cls.REGULARS_COUNT
        ]


class LogFields:
    TIMESTAMP = "timestamp"
    REQUEST = "REQUEST"
    ERROR = "ERROR"
    INFO = "INFO"
    PREDICTION = "PREDICTION"
    FEATURES = "FEATURES"
from pathlib import Path
from datetime import date

DATA_PATH   = Path("/data/feature.csv")
MODEL_DIR   = Path("/models")

# Feature schema
NUM_COLS    = ['user_order_seq', 'ordered_before', 'abandoned_before', 
               'active_snoozed', 'set_as_regular', 'normalised_price', 
               'discount_pct', 'global_popularity', 'count_adults', 
               'count_children', 'count_babies', 'count_pets', 
               'people_ex_baby', 'days_since_purchase_variant_id', 
               'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id', 
               'days_since_purchase_product_type', 'avg_days_to_buy_product_type', 
               'std_days_to_buy_product_type']
CAT_COLS    = ['product_type', 'vendor']

# Model hyper-params (your optimal LightGBM / RF / whatever)
MODEL_PARAMS = {
    "num_leaves": 128,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "min_child_samples": 60,
    "random_state": 55,
}

THRESHOLD   = 0.41          # F0.3 optimum

TODAY_STR   = date.today().strftime("%Y_%m_%d")

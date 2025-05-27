from src.module_4.utils import (
    load_feature_frame,
    validate_event,
    temporal_train_test_split,
)
from src.module_4.config import DATA_PATH
import json
from src.module_4.push_model import PushModel


def handler_fit(event, _):
    try:
        params = validate_event(event, "model_parametrisation")
        filtered_df = load_feature_frame(DATA_PATH)
        X_train, X_test, y_train, y_test = temporal_train_test_split(filtered_df)
        model = PushModel(params).fit(X_train, y_train)

        y_hat_test = model.predict(X_test)
        f03 = model.fbeta_score(y_test, y_hat_test, beta=0.3)

        model_path = model.save()

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "model_path": str(model_path),
                    "val_F0.3": round(f03, 4),
                }
            ),
        }

    except ValueError as e:
        return {"statusCode": "400", "body": json.dumps({"error": str(e)})}

    except Exception as e:
        return {
            "statusCode": "500",
            "body": json.dumps({"error": "An unexpected error occurred: " + str(e)}),
        }

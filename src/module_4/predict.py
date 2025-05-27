import json
from src.module_4.utils import validate_event, get_users_df
from src.module_4.push_model import PushModel


def handler_predict(event, _):
    try:
        users_dict = validate_event(event, "users")
        users_df = get_users_df(users_dict)

        latest_path = PushModel.latest_model_path()
        model = PushModel.load(latest_path)

        preds = model.predict(users_df)
        predictions = dict(zip(users_df.index.astype(str), preds.astype(int)))

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": predictions}),
        }

    except ValueError as ve:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(ve)}),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Unexpected error: {exc}"}),
        }

import pytest
import pandas as pd
from src.module_4.utils import validate_event, get_users_df
from src.module_4.push_model import PushModel


class TestValidateEventInput:
    def test_validate_predict_input_valid(self):
        event = {
            "users": "{"
            '"user1": {"name": "Alice", "feature1": "feat1val1"}, '
            '"user2": {"name": "Bob"}'
            "}"
        }
        result = validate_event(event, "users")
        assert isinstance(result, dict)
        assert "user1" in result
        assert "user2" in result

    def test_validate_predict_input_invalid_json(self):
        event = {
            "users": "{"
            '"user1": {"name": "Alice", "feature1": "feat1val1"}, '
            '"user2": {"name": "Bob"'
        }
        with pytest.raises(ValueError) as exc_info:
            validate_event(event, "users")
        assert str(exc_info.value) == "Invalid JSON format for users."

    def test_validate_predict_input_missing_users(self):
        event = {}
        with pytest.raises(ValueError) as exc_info:
            validate_event(event, "users")
        assert str(exc_info.value) == "users json is required."

    def test_validate_predict_input_non_dict_event(self):
        event = "This is not a dict"
        with pytest.raises(ValueError) as exc_info:
            validate_event(event, "users")
        assert str(exc_info.value) == "Event must be a dictionary."


class TestPushModel:
    def test_model_predict_dummy(self):
        model = PushModel()
        dummy_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
        try:
            result = model.predict_proba(dummy_data)
            assert len(result) == 3
        except Exception as e:
            assert "trained" in str(e).lower() or "fit" in str(e).lower()


class TestGetUsersDF:
    def test_get_users_df_valid(self):
        users = {
            "user1": {"feature1": 1, "feature2": 2},
            "user2": {"feature1": 3, "feature2": 4},
        }
        df = get_users_df(users)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "user1" in df.index
        assert "user2" in df.index

"""
Basic tests for the meteo API module
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.module_1.api_utils import api_call, validate_meteo_api_response
from src.module_1.module_1_meteo_api import get_data_meteo_api, process_data


# Test basic API utility functions
def test_validate_meteo_api_response_valid():
    """Test validation of a valid API response"""
    # Create a sample valid response
    test_data = {
        "daily": {
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "temperature_2m_mean": [10.5, 11.2, 12.0],
            "precipitation_sum": [0, 5.2, 1.3],
            "wind_speed_10m_max": [15.3, 18.2, 10.5],
        },
        "daily_units": {
            "temperature_2m_mean": "°C",
            "precipitation_sum": "mm",
            "wind_speed_10m_max": "km/h",
        },
        "latitude": 40.42,
        "longitude": -3.70,
    }
    expected_variables = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    # Call the function and check result
    result = validate_meteo_api_response(test_data, expected_variables)
    assert result is True


def test_validate_meteo_api_response_missing_keys():
    """Test validation with missing required keys"""
    test_data = {
        "daily": {
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "temperature_2m_mean": [10.5, 11.2, 12.0],
            "precipitation_sum": [0, 5.2, 1.3],
            "wind_speed_10m_max": [15.3, 18.2, 10.5],
        }
        # Missing daily_units, latitude, longitude
    }
    expected_variables = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    result = validate_meteo_api_response(test_data, expected_variables)
    assert result is False


# Test API call function with mocks (no actual network calls)
def test_api_call_success():
    """Test successful API call with mocked response"""
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.json.return_value = {"test": "data"}
    mock_response.status_code = 200

    # Patch the requests.get function to return our mock
    with patch("requests.get", return_value=mock_response):
        result = api_call("http://test-url.com", {"param": "value"})
        # Check that we got the expected result
        assert result == {"test": "data"}


# Test city-specific data retrieval function
def test_get_data_meteo_api_invalid_city():
    """Test get_data_meteo_api with invalid city"""
    result = get_data_meteo_api("InvalidCity")
    assert result is None


# Test data processing function
def test_process_data_empty_input():
    """Test process_data with empty input"""
    result = process_data([])
    assert result.empty


# Test data processing with different frequency values
@pytest.mark.parametrize("freq", ["MS", "D", "h", "W", "BME"])
def test_process_data_with_valid_frequencies(freq):
    """Test process_data with different valid frequency values"""
    # Create sample dataframes
    df = pd.DataFrame(
        {
            "city": ["Madrid", "Madrid", "Madrid"],
            "time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "temperature_2m_mean": [10.5, 11.2, 12.0],
            "precipitation_sum": [0, 5.2, 1.3],
            "wind_speed_10m_max": [15.3, 18.2, 10.5],
        }
    )

    # Process data with the given frequency
    result = process_data([df], freq)

    # Check result is not empty
    assert not result.empty

    # At minimum, we can verify the result has the expected columns
    assert "city" in result.columns
    assert "time" in result.columns
    assert "temperature_2m_mean" in result.columns


# A more complex test using mocks
@patch("src.module_1.module_1_meteo_api.api_call")
@patch("src.module_1.module_1_meteo_api.validate_meteo_api_response")
def test_get_data_meteo_api_success(mock_validate, mock_api_call):
    """Test get_data_meteo_api with successful response"""
    # Create a sample response that mimics the API data
    sample_response = {
        "daily": {
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "temperature_2m_mean": [10.5, 11.2, 12.0],
            "precipitation_sum": [0, 5.2, 1.3],
            "wind_speed_10m_max": [15.3, 18.2, 10.5],
        },
        "daily_units": {
            "temperature_2m_mean": "°C",
            "precipitation_sum": "mm",
            "wind_speed_10m_max": "km/h",
        },
        "latitude": 40.42,
        "longitude": -3.70,
    }

    # Configure the mocks
    mock_api_call.return_value = sample_response
    mock_validate.return_value = True

    # Call the function
    result = get_data_meteo_api("Madrid")

    # Check the result
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "city" in result.columns
    assert "time" in result.columns
    assert "temperature_2m_mean" in result.columns
    assert len(result) == 3
    assert result["city"].iloc[0] == "Madrid"

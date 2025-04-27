from typing import List, Optional
import logging
import requests
from requests.exceptions import RequestException
import time
from models import JSONType

logger = logging.getLogger("meteo-logger")


def api_call(
    url: str, params: JSONType, max_retries: int = 3, backoff_factor: float = 1.0
) -> Optional[JSONType]:
    """
    Makes a generic API call with retry logic and error handling.

    Args:
        url: The API endpoint URL
        params: Dictionary of query parameters
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor to increase wait time between retries

    Returns:
        JSON response data or None if the request failed after retries
    """
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.info(f"Making API request to {url} with params {params}")
            response = requests.get(url, params=params, timeout=10)

            # Check if the request was successful
            response.raise_for_status()

            return response.json()

        except RequestException as e:
            retry_count += 1
            wait_time = backoff_factor * (2 ** (retry_count - 1))

            if response.status_code == 429:  # Too Many Requests
                logger.warning(
                    f"Rate limit hit. Waiting {wait_time} seconds before retry."
                )
            else:
                logger.warning(
                    f"Request failed: {str(e)}. Attempt {retry_count} of {max_retries}."
                )

            if retry_count < max_retries:
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to get data after {max_retries} attempts.")
                return None

    return None


def validate_meteo_api_response(data: JSONType, expected_variables: List[str]) -> bool:
    """
    Validate the API response to ensure it contains expected data structure.

    Args:
        data: The API response data
        expected_variables: List of weather variables we expect to find

    Returns:
        True if the response is valid, False otherwise
    """
    try:
        # Check if required keys exist
        required_keys = ["daily", "daily_units", "latitude", "longitude"]
        if not all(key in data for key in required_keys):
            logger.error(
                f"API response missing required keys. Found: {list(data.keys())}"
            )
            return False

        # Check if all expected variables are in the response
        daily_data = data.get("daily", {})
        if not all(var in daily_data for var in expected_variables):
            logger.error(
                f"""
                API response missing expected variables.
                Found: {list(daily_data.keys())}
                """
            )
            return False

        # Check if time data exists and has the same length as variable data
        if "time" not in daily_data:
            logger.error("API response missing time data")
            return False

        time_length = len(daily_data["time"])
        for var in expected_variables:
            if len(daily_data[var]) != time_length:
                logger.error(f"Variable {var} has inconsistent data length")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating API response: {str(e)}")
        return False

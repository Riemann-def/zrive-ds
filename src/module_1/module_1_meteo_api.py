"""
This module fetches historical weather data from the Open-Meteo API.
It processes data for 3 cities and creates visualizations of temperature,
precipitation, and wind speed trends.
"""

import requests
from requests.exceptions import RequestException
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time

# Configure logging (alternative to "print()" function)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Constants
API_URL = "https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
 "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
 "London": {"latitude": 51.507351, "longitude": -0.127758},
 "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

# Type aliases for better readability
JSONType = Dict[str, Any]
DataFrameType = pd.DataFrame


def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()


def api_call(url: str, params: JSONType, max_retries: int = 3, backoff_factor: float = 1.0) -> Optional[JSONType]:
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
            logger.info(f"Making API request to {url}")
            response = requests.get(url, params=params, timeout=10)
            
            # Check if the request was successful
            response.raise_for_status()
            
            return response.json()
            
        except RequestException as e:
            retry_count += 1
            wait_time = backoff_factor * (2 ** (retry_count - 1))
            
            if response.status_code == 429:  # Too Many Requests
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry.")
            else:
                logger.warning(f"Request failed: {str(e)}. Attempt {retry_count} of {max_retries}.")
                
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
            logger.error(f"API response missing required keys. Found: {list(data.keys())}")
            return False
            
        # Check if all expected variables are in the response
        daily_data = data.get("daily", {})
        if not all(var in daily_data for var in expected_variables):
            logger.error(f"API response missing expected variables. Found: {list(daily_data.keys())}")
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
    
def get_data_meteo_api(
    city: str,
    start_date: str = "2010-01-01",
    end_date: str = "2020-12-31",
    variables: List[str] = None,
) -> Optional[DataFrameType]:
    """
    Fetch weather data for a specific city from the Open-Meteo API.

    Args:
        city: Name of the city (must be a key in the COORDINATES dictionary)
        start_date: Start date for the data in YYYY-MM-DD format
        end_date: End date for the data in YYYY-MM-DD format
        variables: List of weather variables to fetch (defaults to VARIABLES constant)

    Returns:
        DataFrame containing the weather data or None if the request failed
    """

    if variables is None:
        variables = VARIABLES
        
    if city not in COORDINATES:
        logger.error(f"City '{city}' not found in COORDINATES dictionary")
        return None
    
    # Prepare API parameters
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": variables,
        "timezone": "UTC",
    }
    
    # Make API call
    response_data = api_call(API_URL, params)

    if not response_data:
        logger.error(f"Failed to get data for {city}")
        return None
    
    # Validate API response
    if not validate_meteo_api_response(response_data, variables):
        logger.error(f"Invalid API response for {city}")
        return None
    
    logger.info(response_data)
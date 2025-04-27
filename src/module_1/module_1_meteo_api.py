"""
This module fetches historical weather data from the Open-Meteo API.
It processes data for 3 cities and creates visualizations of temperature,
precipitation, and wind speed trends.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from models import JSONType, DataFrameType
from api_utils import api_call, validate_meteo_api_response

# Configure logging (alternative to "print()" function)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("meteo-logger")


# Constants
API_URL = "https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

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
    
    # Convert to DataFrame
    try:
        daily_data = response_data["daily"]
        df = pd.DataFrame(daily_data)
        
        # Convert time strings to datetime objects
        df["time"] = pd.to_datetime(df["time"])
        
        # Add city name for easier identification when combining data
        df["city"] = city
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing data for {city}: {str(e)}")
        return None


def main() -> None:
    """
    Main function to fetch, process, and visualize weather data.
    """
    logger.info("Starting weather data analysis")
    
    # Fetch data for each city
    all_data = []
    for city in COORDINATES:
        logger.info(f"Fetching data for {city}")
        city_data = get_data_meteo_api(city)
        
        if city_data is not None:
            all_data.append(city_data)
        else:
            logger.warning(f"Skipping {city} due to data fetch failure")
    
    if not all_data:
        logger.error("No data available for any city. Exiting.")
        return
    
    # # Process the data (resample to monthly frequency)
    # logger.info("Processing data")
    # processed_data, units = process_data(all_data, resample_freq="MS")
    
    # # Create visualizations
    # logger.info("Creating visualizations")
    # plot_weather_data(processed_data, units)
    
    logger.info("Analysis complete")



if __name__ == "__main__":
    main()

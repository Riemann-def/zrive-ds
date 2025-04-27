"""
This module fetches historical weather data from the Open-Meteo API.
It processes data for 3 cities and creates visualizations of temperature,
precipitation, and wind speed trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from models import JSONType, DataFrameType
from api_utils import api_call, validate_meteo_api_response
from plotting import plot_weather_data
from config import COORDINATES, VARIABLES, API_URL

# Configure logging (alternative to "print()" function)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("meteo-logger")


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


def process_data(
    data_frames: List[DataFrameType], resample_freq: str = "MS"
) -> DataFrameType:
    """
    Process and combine weather data from multiple cities with enhanced analytics.
    Resample the data.

    Args:
        data_frames: List of DataFrames containing weather data for different cities
        resample_freq: Frequency string for resampling (e.g., 'MS' for month start)

    Returns:
        Combined and processed DataFrame with enhanced metrics
    """
    if not data_frames:
        logger.error("No data frames provided for processing")
        return pd.DataFrame()

    # Combine all data frames
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Process each city separately to maintain city distinction
    cities = combined_df["city"].unique()
    processed_dfs = []

    for city in cities:
        city_df = combined_df[combined_df["city"] == city].copy()

        # Set time as index for resampling
        city_df.set_index("time", inplace=True)

        # Resample numeric columns to reduce temporal resolution
        # Use appropriate aggregation methods for each variable
        resampled = pd.DataFrame()

        if "temperature_2m_mean" in city_df.columns:
            resampled["temperature_2m_mean"] = (
                city_df["temperature_2m_mean"].resample(resample_freq).mean()
            )
            # Add min/max temperature for temperature range analysis
            resampled["temperature_2m_min"] = (
                city_df["temperature_2m_mean"].resample(resample_freq).min()
            )
            resampled["temperature_2m_max"] = (
                city_df["temperature_2m_mean"].resample(resample_freq).max()
            )
            resampled["temperature_2m_range"] = (
                resampled["temperature_2m_max"] - resampled["temperature_2m_min"]
            )

        if "precipitation_sum" in city_df.columns:
            resampled["precipitation_sum"] = (
                city_df["precipitation_sum"].resample(resample_freq).sum()
            )
            # Count rainy days (days with precipitation > 1mm)
            resampled["rainy_days"] = (
                (city_df["precipitation_sum"] > 1).resample(resample_freq).sum()
            )
            # Calculate maximum single-day precipitation
            resampled["max_daily_precipitation"] = (
                city_df["precipitation_sum"].resample(resample_freq).max()
            )

        if "wind_speed_10m_max" in city_df.columns:
            resampled["wind_speed_10m_max"] = (
                city_df["wind_speed_10m_max"].resample(resample_freq).max()
            )
            resampled["wind_speed_10m_mean"] = (
                city_df["wind_speed_10m_max"].resample(resample_freq).mean()
            )
            # Count windy days (days with max wind speed > 20km/h)
            resampled["windy_days"] = (
                (city_df["wind_speed_10m_max"] > 20).resample(resample_freq).sum()
            )

        # Calculate moving averages for trend analysis
        if len(resampled) > 3:  # Need at least 3 points for moving average
            for var in VARIABLES:
                if var in resampled.columns:
                    resampled[f"{var}_ma3"] = (
                        resampled[var].rolling(window=3, min_periods=1).mean()
                    )
                    if len(resampled) > 12:  # For annual trends
                        resampled[f"{var}_ma12"] = (
                            resampled[var].rolling(window=12, min_periods=1).mean()
                        )

        # Reset index and add city column back
        resampled.reset_index(inplace=True)
        resampled["city"] = city

        processed_dfs.append(resampled)

    # Combine processed data frames
    result_df = pd.concat(processed_dfs, ignore_index=True)

    # Additional calculations on the combined dataset

    # Calculate ranks for comparison between cities
    for var in VARIABLES:
        if var in result_df.columns:
            # Group by time period and rank cities
            result_df[f"{var}_rank"] = result_df.groupby("time")[var].rank(
                ascending=var != "precipitation_sum"
            )

    return result_df


def main() -> None:
    """
    Main function to load processed data and visualize weather data.
    """

    logger.info("Starting weather data analysis")

    # Fetch data for each city
    all_data = []
    for city in COORDINATES:
        logger.info(f"Fetching data for {city}")
        city_data = get_data_meteo_api(city)

        logger.info(city_data.columns)

        if city_data is not None:
            all_data.append(city_data)
        else:
            logger.warning(f"Skipping {city} due to data fetch failure")

    if not all_data:
        logger.error("No data available for any city. Exiting.")
        return

    # Resample the data (MS: month start frequency, QS: quarter start frequency)
    logger.info("Processing data")
    processed_data = process_data(all_data, resample_freq="QS")

    # Save DataFrame to csv
    try:
        processed_data.to_csv("processed_data_frame.csv")
    except Exception as e:
        logger.warning(f"Failed saving processed DataFrame to csv. Error: {e}")

    # Create visualizations
    logger.info("Creating visualizations")
    plot_weather_data(processed_data)

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()

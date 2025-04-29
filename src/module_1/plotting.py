import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns  # type: ignore
import numpy as np
from typing import List
import logging
import os
from .models import DataFrameType
from .config import VARIABLES, COLORS, UNITS

logger = logging.getLogger("meteo-logger")


def plot_weather_data(
    data: DataFrameType, output_dir: str = "./src/module_1/plots/"
) -> None:
    """
    Create comprehensive visualizations for the weather data.

    Args:
        data: DataFrame containing the processed weather data
        output_dir: Directory where plot images will be saved
    """
    if data.empty:
        logger.error("No data available for plotting")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique cities
    cities_array = data["city"].unique()
    cities = cities_array.tolist()  # Convert numpy array to list

    # 1. Create individual time series plots for each variable
    plot_time_series(data, cities, output_dir + "variables/")

    # 2. Create seasonal analysis plots
    plot_seasonal_analysis(data, cities, output_dir + "seasonal/")

    # 3. Create correlation analysis
    plot_correlation_analysis(data, cities, output_dir + "correlations/")

    # 4. Create distribution plots
    plot_distribution_analysis(data, cities, output_dir + "distributions/")

    # 5. Create multi-panel comparison
    plot_city_comparison(data, cities, output_dir + "cities/")

    logger.info(f"All plots have been saved to {output_dir}")


def plot_time_series(data: DataFrameType, cities: List[str], output_dir: str) -> None:
    """Create time series plots for each weather variable."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for variable in VARIABLES:
        if variable not in data.columns:
            logger.warning(f"Variable {variable} not found in data")
            continue

        plt.figure(figsize=(12, 6))

        for city in cities:
            city_data = data[data["city"] == city]
            plt.plot(
                city_data["time"],
                city_data[variable],
                label=city,
                color=COLORS.get(city, "gray"),
                linewidth=1.5,
            )

        # Add trend lines
        for city in cities:
            city_data = data[data["city"] == city]
            if len(city_data) > 1:  # Need at least 2 points for a line
                # Convert dates to ordinal for linear regression
                x = np.array([d.toordinal() for d in city_data["time"]])
                y = np.array(city_data[variable].values, dtype=float)

                # Calculate trend line using polyfit
                x_np = np.asarray(x, dtype=np.float64)
                y_np = np.asarray(y, dtype=np.float64)
                z = np.polyfit(x_np, y_np, 1)
                p = np.poly1d(z)

                # Plot trend line (dashed)
                plt.plot(
                    city_data["time"],
                    p(x),
                    linestyle="--",
                    color=COLORS.get(city, "gray"),
                    alpha=0.8,
                    linewidth=1.0,
                )

        # Format the plot
        plt.title(f"{variable.replace('_', ' ').title()} (2010-2020)")
        plt.xlabel("Year")
        plt.ylabel(f"{variable.replace('_', ' ').title()} ({UNITS.get(variable, '')})")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

        # Save the plot
        filename = f"{variable}_comparison.png"
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename}")
        logger.info(f"Plot saved as {output_dir}/{filename}")
        plt.close()


def plot_seasonal_analysis(
    data: DataFrameType, cities: List[str], output_dir: str
) -> None:
    """Create seasonal box plots."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Add month and season columns for seasonal analysis
    data_copy = data.copy()
    data_copy["month"] = data_copy["time"].dt.month
    data_copy["year"] = data_copy["time"].dt.year

    # Define seasons (adjusting for southern/northern hemisphere)
    def get_season(row):
        month = row["month"]
        city = row["city"]

        # For Rio (Southern Hemisphere)
        if city == "Rio":
            if month in [12, 1, 2]:
                return "Summer"
            elif month in [3, 4, 5]:
                return "Autumn"
            elif month in [6, 7, 8]:
                return "Winter"
            else:
                return "Spring"
        # For Madrid and London (Northern Hemisphere)
        else:
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

    data_copy["season"] = data_copy.apply(get_season, axis=1)

    # Create seasonal plots for each variable
    for variable in VARIABLES:
        if variable not in data_copy.columns:
            continue

        # Create seasonal boxplots
        plt.figure(figsize=(14, 8))

        for i, city in enumerate(cities):
            city_data = data_copy[data_copy["city"] == city]

            plt.subplot(1, len(cities), i + 1)
            sns.boxplot(
                x="season",
                y=variable,
                hue="season",
                data=city_data,
                palette=["lightblue", "lightgreen", "coral", "khaki"],
                legend=False,
            )

            plt.title(f"{city}: Seasonal Distribution")
            plt.ylabel(
                f"{variable.replace('_', ' ').title()} ({UNITS.get(variable, '')})"
            )
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{variable}_seasonal_boxplots.png")
        plt.close()


def plot_correlation_analysis(
    data: DataFrameType, cities: List[str], output_dir: str
) -> None:
    """Create correlation analysis between different weather variables."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have enough variables to make correlations
    if len(VARIABLES) < 2:
        logger.warning("Not enough variables for correlation analysis")
        return

    # Create correlation plots for each city
    for city in cities:
        city_data = data[data["city"] == city]

        if city_data.empty:
            continue

        # Select only the weather variables for correlation
        corr_data = city_data[VARIABLES].copy()

        # Calculate correlation matrix
        corr_matrix = corr_data.corr()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1
        )

        plt.title(f"{city}: Correlation Between Weather Variables")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{city}_correlation_heatmap.png")
        plt.close()

        # Create scatterplots for each pair of variables
        if len(VARIABLES) >= 2:
            for i, var1 in enumerate(VARIABLES):
                for j, var2 in enumerate(VARIABLES):
                    if i >= j:  # Skip duplicate pairs and self-comparisons
                        continue

                    plt.figure(figsize=(8, 6))
                    plt.scatter(
                        city_data[var1],
                        city_data[var2],
                        alpha=0.6,
                        color=COLORS.get(city, "gray"),
                    )

                    # Add trend line
                    x = city_data[var1].values.astype(float)
                    y = city_data[var2].values.astype(float)

                    # Calculate and plot trend line if we have enough data
                    if len(x) > 1 and not np.isnan(x).all() and not np.isnan(y).all():
                        # Filter out NaN values
                        mask = ~np.isnan(x) & ~np.isnan(y)

                        if np.sum(mask) > 1:  # At least 2 valid points
                            x_valid = x[mask]
                            y_valid = y[mask]

                            z = np.polyfit(x_valid, y_valid, 1)
                            p = np.poly1d(z)

                            x_sorted = np.sort(x_valid)
                            y_trend = p(x_sorted)

                            plt.plot(x_sorted, y_trend, "r--", alpha=0.7)

                    plt.title(
                        f"""
                        {city}: {var1.replace('_', ' ').title()}
                        vs
                        {var2.replace('_', ' ').title()}
                        """
                    )
                    plt.xlabel(
                        f"{var1.replace('_', ' ').title()} ({UNITS.get(var1, '')})"
                    )
                    plt.ylabel(
                        f"{var2.replace('_', ' ').title()} ({UNITS.get(var2, '')})"
                    )
                    plt.grid(True, linestyle="--", alpha=0.7)

                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/{city}_{var1}_vs_{var2}.png")
                    plt.close()


def plot_distribution_analysis(
    data: DataFrameType, cities: List[str], output_dir: str
) -> None:
    """Create distribution analysis plots for each variable."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for variable in VARIABLES:
        if variable not in data.columns:
            continue

        plt.figure(figsize=(12, 6))

        for city in cities:
            city_data = data[data["city"] == city]

            # Plot density distribution
            sns.kdeplot(
                city_data[variable].dropna(),
                label=city,
                color=COLORS.get(city, "gray"),
                fill=True,
                alpha=0.3,
            )

        plt.title(f"Distribution of {variable.replace('_', ' ').title()}")
        plt.xlabel(f"{variable.replace('_', ' ').title()} ({UNITS.get(variable, '')})")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{variable}_distribution.png")
        plt.close()

        # Create boxplot comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x="city",
            y=variable,
            hue="city",
            data=data,
            palette=[COLORS.get(city, "gray") for city in cities],
            legend=False,
        )

        plt.title(f"Distribution of {variable.replace('_', ' ').title()} by City")
        plt.xlabel("City")
        plt.ylabel(f"{variable.replace('_', ' ').title()} ({UNITS.get(variable, '')})")
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{variable}_boxplot.png")
        plt.close()


def plot_city_comparison(
    data: DataFrameType, cities: List[str], output_dir: str
) -> None:
    """Create a multi-panel figure to compare cities across variables."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a multi-panel figure
    fig, axs = plt.subplots(len(cities), len(VARIABLES), figsize=(15, 12), sharex=True)

    for i, city in enumerate(cities):
        city_data = data[data["city"] == city]

        for j, variable in enumerate(VARIABLES):
            # Handle both single and multi-row/column cases
            if len(cities) == 1 and len(VARIABLES) == 1:
                ax = axs
            elif len(cities) == 1:
                ax = axs[j]
            elif len(VARIABLES) == 1:
                ax = axs[i]
            else:
                ax = axs[i, j]

            ax.plot(
                city_data["time"],
                city_data[variable],
                color=COLORS.get(city, "gray"),
                linewidth=1.5,
            )

            # Add rolling average (12-month) if we have enough data
            if len(city_data) > 12:
                # Create copy to avoid SettingWithCopyWarning
                city_data_copy = city_data.copy()
                city_data_copy.set_index("time", inplace=True)
                rolling_avg = (
                    city_data_copy[variable].rolling(window=3, min_periods=1).mean()
                )

                # Plot rolling average as a thicker, semi-transparent line
                ax.plot(
                    rolling_avg.index,
                    rolling_avg.values,
                    color="black",
                    alpha=0.5,
                    linewidth=2.0,
                )

            # Set titles and labels
            if i == 0:
                ax.set_title(variable.replace("_", " ").title())
            if j == 0:
                ax.set_ylabel(f"{city}\n{UNITS.get(variable, '')}")
            if i == len(cities) - 1:
                ax.set_xlabel("Year")

            # Format date axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Show every other year

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/city_comparison.png")
    logger.info(f"Plot saved as {output_dir}/city_comparison.png")
    plt.close()

"""Synthetic data generation module for CrowdCast AI.

NOTICE: This module generates SYNTHETIC DEMONSTRATION DATA for
demonstration, testing, and educational purposes. It does not represent
actual public transport conditions. The architecture is engineered so
that this synthetic dataset can be readily replaced with real-world
automatic passenger counter (APC) or smart-card ticketing logs.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd

from src.preprocessing import (
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
    vectorize_crowd_levels,
)

DATA_DISCLAIMER: str = (
    "SYNTHETIC DEMONSTRATION DATA: Generated for demonstration and modeling evaluation. "
    "Not derived from confidential transit agency telemetry."
)

AVAILABLE_ROUTES = [
    "Route_101",  # Downtown Express (High baseline)
    "Route_102",  # Uptown Metro (High baseline)
    "Route_204",  # Suburban Feeder (Moderate baseline)
    "Route_305",  # Campus University Line (Highly term-dependent)
    "Route_408",  # Crosstown Beltway (Moderate baseline)
    "Route_512",  # Airport & Tech Corridor (Steady baseline)
]

ROUTE_BASE_DEMAND = {
    "Route_101": 22.0,
    "Route_102": 20.0,
    "Route_204": 10.0,
    "Route_305": 14.0,
    "Route_408": 12.0,
    "Route_512": 16.0,
}

DAY_TYPES = ["Weekday", "Weekend", "Holiday"]
WEATHER_CONDITIONS = ["Clear", "Rainy", "Overcast", "Stormy", "Snow/Fog"]


def generate_synthetic_data(
    num_samples: int = 2000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic, reproducible synthetic passenger transport dataset.

    Parameters
    ----------
    num_samples : int, optional
        Number of ride records to generate (default is 2000, exceeding 1000 min requirement).
    random_seed : int, optional
        Fixed seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing features, continuous passenger count, and crowd level.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Feature sampling
    route_ids = rng.choice(AVAILABLE_ROUTES, size=num_samples, p=[0.22, 0.20, 0.16, 0.18, 0.14, 0.10])
    day_types = rng.choice(DAY_TYPES, size=num_samples, p=[0.68, 0.24, 0.08])
    # Departure hours 5 (5:00 AM) to 23 (11:00 PM)
    hours = rng.integers(5, 24, size=num_samples)
    weathers = rng.choice(WEATHER_CONDITIONS, size=num_samples, p=[0.45, 0.25, 0.18, 0.07, 0.05])
    # Nearby events happen ~14% of trips (stadium games, concerts, expos)
    nearby_events = rng.choice([0, 1], size=num_samples, p=[0.86, 0.14])
    # College term is active ~72% of the year
    college_terms = rng.choice([0, 1], size=num_samples, p=[0.28, 0.72])
    # Trip duration between 12 and 80 minutes
    trip_durations = rng.normal(loc=38.0, scale=12.0, size=num_samples)
    trip_durations = np.clip(trip_durations, 12.0, 85.0).round(1)

    # 2. Demand modeling logic with realistic relationships
    passenger_counts = np.zeros(num_samples, dtype=float)

    for i in range(num_samples):
        route = route_ids[i]
        day = day_types[i]
        hour = hours[i]
        weather = weathers[i]
        event = nearby_events[i]
        term = college_terms[i]
        duration = trip_durations[i]

        # Route baseline
        base = ROUTE_BASE_DEMAND[route]

        # Hour and day type dynamics
        time_factor = 0.0
        if day == "Weekday":
            # Morning peak: 7:00 - 9:00 AM
            if 7 <= hour <= 9:
                time_factor += 36.0 + 8.0 * (1.0 - abs(hour - 8.0))
            # Evening peak: 16:00 - 19:00 (4:00 - 7:00 PM)
            elif 16 <= hour <= 19:
                time_factor += 40.0 + 6.0 * (1.0 - abs(hour - 17.5) / 1.5)
            # Midday slump / lunch
            elif 11 <= hour <= 14:
                time_factor += 14.0
            # Early morning (5-6) or late night (21-23)
            elif hour <= 6 or hour >= 21:
                time_factor -= 10.0
            else:
                time_factor += 6.0
        elif day == "Weekend":
            # Weekend peak is shifted to midday and evening recreation (12:00 - 18:00)
            if 12 <= hour <= 18:
                time_factor += 22.0
            elif 19 <= hour <= 22:
                time_factor += 16.0
            elif hour <= 8:
                time_factor -= 14.0
            else:
                time_factor += 8.0
        else:  # Holiday
            if 11 <= hour <= 17:
                time_factor += 12.0
            elif hour <= 8 or hour >= 21:
                time_factor -= 12.0

        # Special route interactions
        if route == "Route_305" and term == 1:
            # Campus line surges during class hours
            if 8 <= hour <= 18:
                base += 24.0

        # Weather impacts
        weather_factor = 0.0
        if weather in ["Rainy", "Snow/Fog"]:
            weather_factor = 9.0  # commuters abandon cycling/walking for buses
        elif weather == "Stormy":
            weather_factor = 14.0  # bunching and heavy transit shelter seeking

        # Event surge
        event_factor = 28.0 if event == 1 else 0.0

        # Trip duration accumulation (longer routes cover more stops)
        duration_factor = (duration - 35.0) * 0.35

        # Stochastic natural variability
        noise = rng.normal(loc=0.0, scale=4.5)

        total_passengers = (
            base
            + time_factor
            + weather_factor
            + event_factor
            + duration_factor
            + noise
        )

        # Realistic transit capacity bounds: minimum 4 passengers, max 115 passengers
        passenger_counts[i] = np.clip(np.round(total_passengers), 4, 115)

    passenger_counts = passenger_counts.astype(int)
    crowd_levels = vectorize_crowd_levels(passenger_counts)

    df = pd.DataFrame(
        {
            "route_id": route_ids,
            "day_type": day_types,
            "departure_hour": hours,
            "weather": weathers,
            "nearby_event": nearby_events,
            "college_term": college_terms,
            "trip_duration": trip_durations,
            "passenger_count": passenger_counts,
            "crowd_level": crowd_levels,
            "data_label": "SYNTHETIC DEMONSTRATION DATA",
        }
    )

    return df


def save_synthetic_data(
    df: pd.DataFrame,
    filepath: str = os.path.join("data", "crowdcast_sample_data.csv"),
) -> str:
    """Save the synthetic dataset to CSV with directory creation."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


def load_dataset(
    filepath: Optional[str] = None,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """Load the dataset from disk or automatically generate if not yet present.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to data/crowdcast_sample_data.csv.
    force_regenerate : bool, optional
        Whether to re-generate synthetic data even if file exists.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "crowdcast_sample_data.csv",
        )

    if force_regenerate or not os.path.exists(filepath):
        df = generate_synthetic_data(num_samples=2000, random_seed=42)
        save_synthetic_data(df, filepath)
        return df

    return pd.read_csv(filepath)


if __name__ == "__main__":
    df = generate_synthetic_data(num_samples=2000, random_seed=42)
    path = save_synthetic_data(df)
    print(f"Generated {len(df)} rows. Saved to: {path}")
    print(df["crowd_level"].value_counts())

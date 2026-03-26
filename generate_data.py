"""
Synthetic Data Generator for Flink Delivery Promise Optimization
================================================================
Generates realistic quick-commerce delivery data mimicking Flink's operations.
Each row represents a single delivery order with operational, environmental,
and temporal features.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

N_ORDERS = 15_000
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)

# --- City warehouse configs (German cities where Flink operates) ---
WAREHOUSES = {
    "BER-01": {"city": "Berlin", "lat": 52.52, "lon": 13.405, "capacity": 80, "avg_riders": 12},
    "BER-02": {"city": "Berlin", "lat": 52.48, "lon": 13.44, "capacity": 60, "avg_riders": 9},
    "HAM-01": {"city": "Hamburg", "lat": 53.55, "lon": 9.99, "capacity": 70, "avg_riders": 10},
    "MUC-01": {"city": "Munich", "lat": 48.14, "lon": 11.58, "capacity": 75, "avg_riders": 11},
    "CGN-01": {"city": "Cologne", "lat": 50.94, "lon": 6.96, "capacity": 55, "avg_riders": 8},
    "DUS-01": {"city": "Düsseldorf", "lat": 51.23, "lon": 6.78, "capacity": 50, "avg_riders": 7},
}

PRODUCT_CATEGORIES = [
    "fresh_produce", "dairy", "beverages", "snacks", "frozen",
    "household", "personal_care", "bakery", "meat_fish", "alcohol"
]


def random_timestamps(n: int) -> pd.Series:
    """Generate realistic order timestamps with daily/weekly seasonality."""
    dates = []
    for _ in range(n):
        day_offset = np.random.randint(0, (END_DATE - START_DATE).days)
        date = START_DATE + timedelta(days=day_offset)

        # Hour distribution: peaks at lunch (12-14) and evening (18-21)
        hour_weights = np.array([
            0.5, 0.3, 0.2, 0.1, 0.1, 0.2,   # 0-5
            0.5, 1.0, 2.0, 3.0, 4.0, 5.0,     # 6-11
            7.0, 6.0, 4.0, 3.5, 4.0, 6.0,     # 12-17
            8.0, 9.0, 8.0, 6.0, 4.0, 2.0      # 18-23
        ])
        hour_weights /= hour_weights.sum()
        hour = np.random.choice(24, p=hour_weights)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)

        ts = date.replace(hour=hour, minute=minute, second=second)
        dates.append(ts)
    return pd.Series(dates)


def generate_weather(timestamps: pd.Series) -> pd.DataFrame:
    """Generate correlated weather features based on month and randomness."""
    n = len(timestamps)
    months = timestamps.dt.month

    # Temperature: seasonal pattern (German climate)
    base_temp = 5 + 12 * np.sin((months - 4) * np.pi / 6)
    temperature_c = base_temp + np.random.normal(0, 3, n)

    # Precipitation probability higher in autumn/winter
    precip_base = 0.25 + 0.15 * np.cos((months - 11) * np.pi / 6)
    is_raining = np.random.random(n) < precip_base
    rain_intensity_mm = np.where(is_raining, np.random.exponential(3, n), 0)

    # Wind speed
    wind_speed_kmh = np.random.gamma(2, 5, n) + 3

    return pd.DataFrame({
        "temperature_c": np.round(temperature_c, 1),
        "is_raining": is_raining.astype(int),
        "rain_intensity_mm": np.round(rain_intensity_mm, 1),
        "wind_speed_kmh": np.round(wind_speed_kmh, 1),
    })


def generate_orders(n: int) -> pd.DataFrame:
    """Generate the full synthetic order dataset."""

    # --- Timestamps ---
    timestamps = random_timestamps(n).sort_values().reset_index(drop=True)

    # --- Warehouse assignment ---
    wh_ids = list(WAREHOUSES.keys())
    wh_weights = [WAREHOUSES[w]["capacity"] for w in wh_ids]
    wh_weights = np.array(wh_weights) / sum(wh_weights)
    warehouse_id = np.random.choice(wh_ids, n, p=wh_weights)

    # --- Order characteristics ---
    n_items = np.random.poisson(4, n) + 1  # at least 1 item
    n_items = np.clip(n_items, 1, 25)

    # Basket value correlated with n_items
    basket_value_eur = n_items * np.random.uniform(1.5, 4.5, n) + np.random.normal(0, 2, n)
    basket_value_eur = np.clip(basket_value_eur, 2.5, 120).round(2)

    # Heavy items flag (beverages, household)
    has_heavy_items = np.random.random(n) < 0.3

    # Primary product category
    primary_category = np.random.choice(PRODUCT_CATEGORIES, n)

    # --- Operational features ---
    hour = timestamps.dt.hour.values
    day_of_week = timestamps.dt.dayofweek.values

    # Active orders in warehouse (higher during peak hours)
    peak_multiplier = np.where(
        ((hour >= 12) & (hour <= 14)) | ((hour >= 18) & (hour <= 21)),
        np.random.uniform(1.5, 2.5, n),
        np.random.uniform(0.5, 1.2, n)
    )
    # Weekend boost
    weekend_boost = np.where(day_of_week >= 5, 1.3, 1.0)

    warehouse_capacity = np.array([WAREHOUSES[w]["capacity"] for w in warehouse_id])
    active_orders = (warehouse_capacity * 0.4 * peak_multiplier * weekend_boost + np.random.normal(0, 3, n))
    active_orders = np.clip(active_orders, 1, warehouse_capacity).astype(int)

    warehouse_utilization = (active_orders / warehouse_capacity).round(3)

    # Rider availability
    avg_riders = np.array([WAREHOUSES[w]["avg_riders"] for w in warehouse_id])
    available_riders = (avg_riders * np.random.uniform(0.3, 1.0, n) * peak_multiplier * 0.6).astype(int)
    available_riders = np.clip(available_riders, 0, avg_riders * 2)
    rider_utilization = np.where(
        available_riders > 0,
        (active_orders / (available_riders + 1)).round(3),
        999  # no riders available -> extreme pressure
    )

    # Distance to customer (km) — delivery radius ~1-4 km for quick commerce
    distance_km = np.random.gamma(2.5, 0.7, n)
    distance_km = np.clip(distance_km, 0.3, 5.5).round(2)

    # --- Weather ---
    weather = generate_weather(timestamps)

    # --- Pick & pack time (minutes) ---
    # Base time depends on items + warehouse utilization
    pick_time_min = (
        1.5  # base
        + 0.4 * n_items
        + 3.0 * warehouse_utilization
        + 1.0 * has_heavy_items
        + np.random.normal(0, 0.8, n)
    )
    pick_time_min = np.clip(pick_time_min, 1.0, 20.0).round(1)

    # --- Rider travel time (minutes) ---
    # Affected by distance, weather, wind, traffic (hour-based)
    traffic_factor = np.where(
        ((hour >= 8) & (hour <= 9)) | ((hour >= 17) & (hour <= 19)),
        np.random.uniform(1.2, 1.6, n),  # rush hour
        np.random.uniform(0.8, 1.1, n)
    )
    rain_slow = 1.0 + 0.15 * weather["is_raining"].values + 0.02 * weather["rain_intensity_mm"].values
    wind_slow = 1.0 + 0.005 * weather["wind_speed_kmh"].values

    travel_time_min = (
        distance_km * 3.2  # ~19 km/h avg cycling speed -> ~3.2 min/km
        * traffic_factor
        * rain_slow
        * wind_slow
        + np.random.normal(0, 1.2, n)
    )
    travel_time_min = np.clip(travel_time_min, 1.5, 35.0).round(1)

    # --- Rider wait time (if no rider available immediately) ---
    rider_wait_min = np.where(
        available_riders <= 1,
        np.random.exponential(4, n),
        np.where(
            available_riders <= 3,
            np.random.exponential(1.5, n),
            np.random.exponential(0.3, n)
        )
    )
    rider_wait_min = np.clip(rider_wait_min, 0, 15).round(1)

    # === TARGET: Total delivery time (minutes) ===
    delivery_time_min = (pick_time_min + rider_wait_min + travel_time_min).round(1)

    # Add some noise for real-world variability (unexpected delays)
    random_delay = np.where(
        np.random.random(n) < 0.08,  # 8% chance of unusual delay
        np.random.uniform(3, 12, n),
        0
    )
    delivery_time_min = (delivery_time_min + random_delay).round(1)
    delivery_time_min = np.clip(delivery_time_min, 3, 60)

    # --- Promised delivery time (what customer was told) ---
    # Current naive system: fixed buckets
    promised_time_min = np.where(
        delivery_time_min < 12, 10,
        np.where(delivery_time_min < 20, 15,
        np.where(delivery_time_min < 30, 25, 35))
    )

    # Whether promise was kept
    promise_kept = (delivery_time_min <= promised_time_min).astype(int)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "order_id": [f"ORD-{i:06d}" for i in range(n)],
        "timestamp": timestamps,
        "warehouse_id": warehouse_id,
        "city": [WAREHOUSES[w]["city"] for w in warehouse_id],

        # Temporal
        "hour": hour,
        "day_of_week": day_of_week,
        "month": timestamps.dt.month.values,
        "is_weekend": (day_of_week >= 5).astype(int),

        # Order
        "n_items": n_items,
        "basket_value_eur": basket_value_eur,
        "has_heavy_items": has_heavy_items.astype(int),
        "primary_category": primary_category,

        # Operational
        "active_orders": active_orders,
        "warehouse_utilization": warehouse_utilization,
        "available_riders": available_riders,
        "rider_utilization": rider_utilization.round(3),
        "distance_km": distance_km,

        # Weather
        "temperature_c": weather["temperature_c"],
        "is_raining": weather["is_raining"],
        "rain_intensity_mm": weather["rain_intensity_mm"],
        "wind_speed_kmh": weather["wind_speed_kmh"],

        # Sub-components (for analysis, not model input)
        "pick_time_min": pick_time_min,
        "rider_wait_min": rider_wait_min,
        "travel_time_min": travel_time_min,

        # Target
        "delivery_time_min": delivery_time_min,
        "promised_time_min": promised_time_min,
        "promise_kept": promise_kept,
    })

    return df


if __name__ == "__main__":
    print("Generating synthetic Flink delivery data...")
    df = generate_orders(N_ORDERS)

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "flink_deliveries.csv")
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} orders -> {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Promise kept rate: {df['promise_kept'].mean():.1%}")
    print(f"Avg delivery time: {df['delivery_time_min'].mean():.1f} min")
    print(f"Delivery time range: {df['delivery_time_min'].min():.1f} - {df['delivery_time_min'].max():.1f} min")
    print(f"\nWarehouses: {df['warehouse_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

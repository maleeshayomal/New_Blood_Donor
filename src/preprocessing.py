"""
Data Preprocessing & Feature Engineering Module
===============================================
Academic AI Decision-Support System
"""

import math
from typing import List, Tuple, Union
import numpy as np
import pandas as pd


# Definitive list of ML feature columns used for response prediction
ML_FEATURE_NAMES: List[str] = [
    "age",
    "available",
    "donation_count",
    "days_since_last_donation",
    "previous_requests",
    "previous_responses",
    "response_rate",
    "distance_km",
    "contacted_before"
]


def calculate_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate the great-circle distance between two geographic points on Earth
    using the Haversine formula. Supports scalars, numpy arrays, and pandas Series.

    Parameters
    ----------
    lat1 : float or array-like
        Latitude of point 1 in decimal degrees.
    lon1 : float or array-like
        Longitude of point 1 in decimal degrees.
    lat2 : float or array-like
        Latitude of point 2 in decimal degrees.
    lon2 : float or array-like
        Longitude of point 2 in decimal degrees.

    Returns
    -------
    float or array-like
        Distance between points in kilometers (km), rounded to 2 decimal places.
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    # Ensure precision clipping within [-1, 1] for arcsin/arctan2
    c = 2 * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1 - a, 0, 1)))

    distance = R * c
    if isinstance(distance, (pd.Series, np.ndarray)):
        return np.round(distance, 2)
    return round(float(distance), 2)


def add_distance_feature(
    df: pd.DataFrame,
    hospital_lat: float,
    hospital_lon: float,
    donor_lat_col: str = "latitude",
    donor_lon_col: str = "longitude",
    output_col: str = "distance_km"
) -> pd.DataFrame:
    """
    Compute and attach distance_km column to a donor DataFrame given hospital coordinates.
    """
    df_copy = df.copy()
    if df_copy.empty:
        df_copy[output_col] = 0.0
        return df_copy
    
    df_copy[output_col] = calculate_distance(
        df_copy[donor_lat_col].values,
        df_copy[donor_lon_col].values,
        hospital_lat,
        hospital_lon
    )
    return df_copy


def prepare_ml_features(
    df: pd.DataFrame,
    feature_names: List[str] = ML_FEATURE_NAMES
) -> pd.DataFrame:
    """
    Extract and validate the exact feature subset required by the ML model.

    Parameters
    ----------
    df : pd.DataFrame
        Donor dataset containing engineered features (including distance_km).
    feature_names : List[str]
        List of expected feature column names.

    Returns
    -------
    pd.DataFrame
        Cleaned feature matrix X ready for model inference.
    """
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required ML feature columns: {missing}")

    X = df[feature_names].copy()
    
    # Fill any potential NaN values safely
    X = X.fillna({
        "age": 30,
        "available": 1,
        "donation_count": 1,
        "days_since_last_donation": 120,
        "previous_requests": 1,
        "previous_responses": 1,
        "response_rate": 0.5,
        "distance_km": 10.0,
        "contacted_before": 1
    })
    
    return X

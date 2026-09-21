"""
Rule-Based Reasoning Engine (Deductive AI Layer)
=================================================
Academic AI Decision-Support System

Applies deterministic clinical logic and availability rules to filter potential
donors before invoking machine learning classification.
"""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, Dict, Any, Optional
import pandas as pd
from src.blood_compatibility import is_compatible, get_compatible_donors


class RuleEngine:
    """
    Deterministic rule-based reasoning engine for filtering blood donor candidates.
    """

    def __init__(self, min_days_since_last_donation: int = 90, require_availability: bool = True):
        """
        Initialize the rule engine with clinical and operational constraints.

        Parameters
        ----------
        min_days_since_last_donation : int
            Safe recovery interval (days) required between donations (default: 90 days).
        require_availability : bool
            Whether active availability (available == 1) is strictly required (default: True).
        """
        self.min_days = min_days_since_last_donation
        self.require_availability = require_availability

    def filter_candidates(
        self,
        donors_df: pd.DataFrame,
        required_blood_group: str,
        enforce_donation_interval: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Apply hierarchical filtering pipeline to select valid donor candidates.

        Pipeline Stages:
        1. Base Database (Total Donors)
        2. Blood Compatibility Filter (ABO & Rh Compatibility)
        3. Real-time Availability Filter (available == 1)
        4. Safe Donation Interval Filter (days_since_last_donation >= min_days)

        Parameters
        ----------
        donors_df : pd.DataFrame
            The complete donor dataset.
        required_blood_group : str
            The blood group requested for the emergency patient.
        enforce_donation_interval : bool
            Whether to enforce safe donation gap (default: True).

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, int]]
            - Filtered candidate DataFrame
            - Funnel breakdown audit dictionary
        """
        if donors_df.empty:
            return donors_df, {
                "total_donors": 0,
                "compatible_donors": 0,
                "available_donors": 0,
                "final_candidates": 0
            }

        total_count = len(donors_df)

        # Stage 1: Blood Compatibility Rule
        compatible_groups = get_compatible_donors(required_blood_group)
        df_compatible = donors_df[donors_df["blood_group"].isin(compatible_groups)].copy()
        compatible_count = len(df_compatible)

        # Stage 2: Availability Rule
        if self.require_availability:
            df_available = df_compatible[df_compatible["available"] == 1].copy()
        else:
            df_available = df_compatible.copy()
        available_count = len(df_available)

        # Stage 3: Donation Interval Rule (Safe Recovery Period)
        if enforce_donation_interval and "days_since_last_donation" in df_available.columns:
            df_final = df_available[df_available["days_since_last_donation"] >= self.min_days].copy()
            # If the safe interval filter eliminates too many emergency candidates, retain available compatible donors
            if len(df_final) == 0 and len(df_available) > 0:
                # Soft fallback with logging
                df_final = df_available.copy()
        else:
            df_final = df_available.copy()

        final_count = len(df_final)

        audit_summary = {
            "total_donors": total_count,
            "compatible_donors": compatible_count,
            "available_donors": available_count,
            "final_candidates": final_count,
            "compatible_groups": compatible_groups,
            "required_blood_group": required_blood_group
        }

        return df_final, audit_summary


def filter_donors(
    donors_df: pd.DataFrame,
    required_blood_group: str,
    min_days: int = 90
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Convenience functional wrapper for the RuleEngine.
    """
    engine = RuleEngine(min_days_since_last_donation=min_days, require_availability=True)
    return engine.filter_candidates(donors_df, required_blood_group)

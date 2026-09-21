"""
Intelligent Recommendation, Ranking & Explainable AI (XAI) Module
==================================================================
Academic AI Decision-Support System

Calculates transparent multi-criteria recommendation scores, assigns recommendation
tiers, ranks candidates, and generates dynamic evidence-based explanations.
"""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from src.blood_compatibility import get_compatibility_description


# Default Configurable Weights
DEFAULT_WEIGHTS: Dict[str, float] = {
    "response_probability": 0.40,
    "availability": 0.30,
    "proximity": 0.20,
    "donation_history": 0.10
}

# Default Configurable Recommendation Thresholds
DEFAULT_TIER_THRESHOLDS: Dict[str, float] = {
    "Highly Recommended": 90.0,
    "Recommended": 75.0,
    "Moderately Recommended": 60.0,
    "Low Priority": 0.0
}


class DonorRanker:
    """
    Multi-criteria ranking and explainability engine for blood donor recommendations.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.thresholds = thresholds or DEFAULT_TIER_THRESHOLDS.copy()

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-3):
            # Normalize automatically
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def calculate_component_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate normalized (0-100) sub-scores for each recommendation criterion.
        """
        df_scored = df.copy()

        # 1. Response Probability Score (0 - 100)
        df_scored["score_response_prob"] = np.round(df_scored["response_probability"] * 100.0, 1)

        # 2. Availability Score (0 or 100)
        df_scored["score_availability"] = np.where(df_scored["available"] == 1, 100.0, 0.0)

        # 3. Proximity Score (0 - 100): closer donors get higher scores
        # Formula: max(0, 100 - distance_km * 10)
        raw_proximity = 100.0 - (df_scored["distance_km"] * 10.0)
        df_scored["score_proximity"] = np.round(np.clip(raw_proximity, 0.0, 100.0), 1)

        # 4. Donation History Score (0 - 100): normalized by donation count (e.g. 10+ donations = 100)
        raw_history = df_scored["donation_count"] * 10.0
        df_scored["score_donation_history"] = np.round(np.clip(raw_history, 0.0, 100.0), 1)

        # Final Weighted Recommendation Score (0 - 100)
        df_scored["recommendation_score"] = np.round(
            (self.weights["response_probability"] * df_scored["score_response_prob"]) +
            (self.weights["availability"] * df_scored["score_availability"]) +
            (self.weights["proximity"] * df_scored["score_proximity"]) +
            (self.weights["donation_history"] * df_scored["score_donation_history"]),
            1
        )

        # Assign Recommendation Tier Label
        df_scored["recommendation_label"] = df_scored["recommendation_score"].apply(self._assign_tier_label)

        return df_scored

    def _assign_tier_label(self, score: float) -> str:
        """Categorize recommendation score into transparent tier labels."""
        if score >= self.thresholds.get("Highly Recommended", 90.0):
            return "Highly Recommended"
        elif score >= self.thresholds.get("Recommended", 75.0):
            return "Recommended"
        elif score >= self.thresholds.get("Moderately Recommended", 60.0):
            return "Moderately Recommended"
        else:
            return "Low Priority"

    def rank_candidates(
        self,
        df: pd.DataFrame,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Score and rank candidate donors in descending order.

        Parameters
        ----------
        df : pd.DataFrame
            Candidate donors with predicted response probabilities and distances.
        top_n : int
            Number of top candidates to return (e.g., 3 or 5).

        Returns
        -------
        pd.DataFrame
            Ranked donor DataFrame with score breakdowns and assigned ranks.
        """
        if df.empty:
            return df.copy()

        scored_df = self.calculate_component_scores(df)

        # Multi-level sort: score (desc), response_prob (desc), distance (asc)
        sorted_df = scored_df.sort_values(
            by=["recommendation_score", "response_probability", "distance_km"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        sorted_df["rank"] = range(1, len(sorted_df) + 1)

        if top_n is not None and top_n > 0:
            return sorted_df.head(top_n)
        return sorted_df


def generate_donor_explanation(
    donor_row: pd.Series,
    required_blood_group: str,
    hospital_name: str = "Hospital"
) -> Dict[str, Any]:
    """
    Generate dynamic, evidence-based explanations derived directly from feature values.

    Parameters
    ----------
    donor_row : pd.Series
        Individual donor record with computed scores.
    required_blood_group : str
        The requested blood group.
    hospital_name : str
        The destination medical center.

    Returns
    -------
    Dict[str, Any]
        Structured explanation rationale and summary bullet points.
    """
    bullets = []
    
    # 1. Compatibility rationale
    compat_desc = get_compatibility_description(donor_row["blood_group"], required_blood_group)
    bullets.append(f"✓ {compat_desc}")

    # 2. Availability rationale
    if donor_row["available"] == 1:
        bullets.append("✓ Currently active and marked available for emergency dispatch")
    else:
        bullets.append("⚠ Currently marked unavailable (requires special verification)")

    # 3. Proximity rationale
    dist = donor_row.get("distance_km", 0.0)
    if dist <= 3.0:
        bullets.append(f"✓ Exceptional proximity: only {dist} km from {hospital_name}")
    elif dist <= 10.0:
        bullets.append(f"✓ Rapid transit range: {dist} km from {hospital_name}")
    else:
        bullets.append(f"• Regional distance: {dist} km from {hospital_name}")

    # 4. Historical reliability rationale
    resp_rate = donor_row.get("response_rate", 0.0) * 100
    prev_resp = donor_row.get("previous_responses", 0)
    prev_req = donor_row.get("previous_requests", 0)
    if resp_rate >= 80:
        bullets.append(f"✓ Proven high response reliability ({resp_rate:.0f}% rate across {prev_req} past requests)")
    elif resp_rate >= 50:
        bullets.append(f"✓ Moderate historical responsiveness ({resp_rate:.0f}% across {prev_req} requests)")
    else:
        bullets.append(f"• Historical response record: {resp_rate:.0f}% ({prev_resp}/{prev_req})")

    # 5. ML Predicted likelihood
    prob_pct = donor_row.get("response_probability", 0.0) * 100
    bullets.append(f"✓ {prob_pct:.1f}% ML-predicted probability of affirmative emergency response")

    # 6. Donation frequency & safe interval
    donations = donor_row.get("donation_count", 0)
    days_since = donor_row.get("days_since_last_donation", 0)
    bullets.append(f"✓ Experienced donor ({donations} prior donations; {days_since} days since last donation)")

    return {
        "donor_id": donor_row.get("donor_id", "Unknown"),
        "blood_group": donor_row.get("blood_group", ""),
        "rank": int(donor_row.get("rank", 1)),
        "score": float(donor_row.get("recommendation_score", 0.0)),
        "label": str(donor_row.get("recommendation_label", "")),
        "bullets": bullets,
        "component_breakdown": {
            "Response Probability Score (40%)": float(donor_row.get("score_response_prob", 0.0)),
            "Availability Score (30%)": float(donor_row.get("score_availability", 0.0)),
            "Proximity Score (20%)": float(donor_row.get("score_proximity", 0.0)),
            "Donation History Score (10%)": float(donor_row.get("score_donation_history", 0.0))
        }
    }

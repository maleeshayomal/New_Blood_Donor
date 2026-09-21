"""
Machine Learning Inference & Prediction Module
==============================================
Academic AI Decision-Support System

Loads the serialized classification model and predicts response probability for
candidate donors based on emergency location context.
"""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import ML_FEATURE_NAMES, add_distance_feature, prepare_ml_features
from src.train_model import run_training_pipeline


class DonorPredictor:
    """
    Inference engine for estimating donor response likelihood.
    """

    def __init__(self, model_path: str = "models/donor_response_model.pkl"):
        self.model_path = model_path
        self.model = self._load_or_train_model()

    def _load_or_train_model(self):
        """Load trained model from disk or train fresh if not yet created."""
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                print(f"Warning: Could not load model from {self.model_path}: {e}. Retraining...")
        
        # Train and save
        _, model = run_training_pipeline()
        return model

    def predict_response_probabilities(
        self,
        candidates_df: pd.DataFrame,
        hospital_lat: float,
        hospital_lon: float
    ) -> pd.DataFrame:
        """
        Calculate distance to hospital and predict response probabilities for candidates.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            Rule-filtered candidate donors.
        hospital_lat : float
            Emergency hospital latitude coordinate.
        hospital_lon : float
            Emergency hospital longitude coordinate.

        Returns
        -------
        pd.DataFrame
            Candidates DataFrame with added 'distance_km' and 'response_probability' columns.
        """
        if candidates_df.empty:
            candidates_copy = candidates_df.copy()
            candidates_copy["distance_km"] = 0.0
            candidates_copy["response_probability"] = 0.0
            return candidates_copy

        # 1. Calculate dynamic distance to the emergency hospital
        df_with_dist = add_distance_feature(
            candidates_df,
            hospital_lat=hospital_lat,
            hospital_lon=hospital_lon
        )

        # 2. Extract ML feature matrix
        X = prepare_ml_features(df_with_dist, ML_FEATURE_NAMES)

        # 3. Model inference (predict_proba)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            # Fallback for models without predict_proba
            preds = self.model.predict(X)
            probs = preds.astype(float)

        # Clip and round
        probs_clean = np.clip(probs, 0.01, 0.99)
        df_with_dist["response_probability"] = np.round(probs_clean, 4)

        return df_with_dist


def predict_candidates(
    candidates_df: pd.DataFrame,
    hospital_lat: float,
    hospital_lon: float,
    model_path: str = "models/donor_response_model.pkl"
) -> pd.DataFrame:
    """
    Functional convenience wrapper for response probability prediction.
    """
    predictor = DonorPredictor(model_path=model_path)
    return predictor.predict_response_probabilities(candidates_df, hospital_lat, hospital_lon)

import sys
import os
import io

# Set UTF-8 encoding for standard output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.blood_compatibility import get_compatible_donors, is_compatible
from src.rule_engine import filter_donors
from src.prediction import predict_candidates
from src.ranking import DonorRanker, generate_donor_explanation
from src.database import BloodDonorDatabase


def test_emergency_workflow():
    print("Testing End-to-End Emergency Recommendation Pipeline...")
    
    # 1. Load data
    db = BloodDonorDatabase("data/blood_donor.db")
    all_donors = db.get_all_donors()
    print(f"1. Total donors in DB: {len(all_donors)}")
    assert len(all_donors) >= 1200, f"Expected >= 1200 donors, found {len(all_donors)}"

    # 2. Emergency Request inputs
    req_blood = "O+"
    hospital_name = "National Hospital Colombo"
    hosp_lat, hosp_lon = 6.9271, 79.8612

    # 3. Rule-based Reasoning
    candidates, audit = filter_donors(all_donors, req_blood, min_days=90)
    print(f"2. Rule Engine Funnel: {audit}")
    assert len(candidates) > 0, "Candidates should not be empty for O+"

    # 4. ML Prediction
    predicted_df = predict_candidates(candidates, hosp_lat, hosp_lon)
    print(f"3. Predicted probabilities for {len(predicted_df)} candidates. Sample: {predicted_df['response_probability'].head().tolist()}")
    assert "response_probability" in predicted_df.columns
    assert "distance_km" in predicted_df.columns

    # 5. Ranking & Recommendation
    ranker = DonorRanker()
    top_5 = ranker.rank_candidates(predicted_df, top_n=5)
    print("\n" + "="*80)
    print(f"{'TOP RECOMMENDED DONORS FOR O+ AT COLOMBO':^80}")
    print("="*80)
    print(f"{'Rank':<5} | {'Donor ID':<9} | {'Blood':<6} | {'Distance':<10} | {'Avail':<6} | {'Prob':<8} | {'Score':<6} | {'Tier'}")
    print("-" * 80)
    for _, row in top_5.iterrows():
        print(f"{row['rank']:<5} | {row['donor_id']:<9} | {row['blood_group']:<6} | {row['distance_km']} km    | {row['available']:<6} | {row['response_probability']*100:.1f}%   | {row['recommendation_score']:<6} | {row['recommendation_label']}")

    # 6. Explainability check
    top_donor = top_5.iloc[0]
    explanation = generate_donor_explanation(top_donor, req_blood, hospital_name)
    print("\n" + "-"*80)
    print(f"EXPLANATION FOR TOP DONOR ({top_donor['donor_id']}):")
    for b in explanation["bullets"]:
        print(" ", b)
    print("-" * 80)
    
    print("\n Pipeline Verification Passed Successfully!")


if __name__ == "__main__":
    test_emergency_workflow()

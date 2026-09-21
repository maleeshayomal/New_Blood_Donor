"""
AI-Based Emergency Blood Donor Recommendation System
=====================================================
Interactive Streamlit Clinical Decision-Support Web Application

Pillars:
1. Rule-Based Reasoning (ABO & Rh Compatibility, Active Availability, Safe Interval)
2. Machine Learning Classification (Response Likelihood Estimation)
3. Multi-Criteria Recommendation Scoring & Explainable AI (XAI)
"""

import os
import sys
import datetime
from typing import Dict, Any, List

# Ensure project root is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Core System Modules
from src.blood_compatibility import (
    RECIPIENT_TO_DONOR_COMPATIBILITY,
    ALL_BLOOD_GROUPS,
    get_compatible_donors,
    get_compatibility_description
)
from src.rule_engine import filter_donors, RuleEngine
from src.preprocessing import calculate_distance
from src.prediction import DonorPredictor
from src.ranking import DonorRanker, generate_donor_explanation, DEFAULT_WEIGHTS, DEFAULT_TIER_THRESHOLDS
from src.database import BloodDonorDatabase
from src.data_generator import CITY_COORDINATES, save_and_init_db
from src.train_model import run_training_pipeline

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Emergency Blood Donor Recommender",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Styling (CSS)
st.markdown("""
<style>
    /* Main container and font styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #8B0000 0%, #C0392B 60%, #E74C3C 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 25px rgba(192, 57, 43, 0.25);
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 15px;
        opacity: 0.92;
        margin-top: 6px;
        margin-bottom: 0;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #edf2f7;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #c0392b;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Disclaimer Alert */
    .disclaimer-banner {
        background-color: #fff8e1;
        border-left: 5px solid #ffa000;
        padding: 12px 18px;
        border-radius: 8px;
        font-size: 13px;
        color: #5d4037;
        margin-bottom: 20px;
    }

    /* Recommendation Badges */
    .tier-badge-high {
        background-color: #d4edda;
        color: #155724;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
    }
    .tier-badge-rec {
        background-color: #cce5ff;
        color: #004085;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
    }
    .tier-badge-mod {
        background-color: #fff3cd;
        color: #856404;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
    }
    .tier-badge-low {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
    }
    
    /* Funnel Step Card */
    .funnel-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .funnel-number {
        font-size: 24px;
        font-weight: 800;
        color: #2b6cb0;
    }
    .funnel-title {
        font-size: 12px;
        color: #4a5568;
        font-weight: 600;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Database & Model Caching Utilities
# -----------------------------------------------------------------------------
@st.cache_resource
def get_database() -> BloodDonorDatabase:
    db_path = "data/blood_donor.db"
    if not os.path.exists(db_path) or not os.path.exists("data/donors.csv"):
        save_and_init_db()
    return BloodDonorDatabase(db_path)


@st.cache_resource
def get_predictor() -> DonorPredictor:
    model_path = "models/donor_response_model.pkl"
    if not os.path.exists(model_path):
        run_training_pipeline()
    return DonorPredictor(model_path)


@st.cache_data
def get_model_metrics() -> Dict[str, Any]:
    metrics_path = "models/model_metrics.pkl"
    if not os.path.exists(metrics_path):
        results, _ = run_training_pipeline()
        return results
    try:
        return joblib.load(metrics_path)
    except Exception:
        results, _ = run_training_pipeline()
        return results


# Preset Hospital Directory for Instant Selection
PRESET_HOSPITALS = {
    "National Hospital of Sri Lanka (Colombo)": (6.9271, 79.8612),
    "Colombo South Teaching Hospital (Kalubowila)": (6.8770, 79.8820),
    "Teaching Hospital Kandy": (7.2906, 80.6337),
    "District General Hospital Gampaha": (7.0840, 80.0098),
    "Teaching Hospital Karapitiya (Galle)": (6.0329, 80.2168),
    "District General Hospital Negombo": (7.2083, 79.8358),
    "Teaching Hospital Kurunegala": (7.4863, 80.3623),
    "District General Hospital Matara": (5.9549, 80.5550),
    "Teaching Hospital Jaffna": (9.6615, 80.0255),
    "Custom Location (Enter Coordinates)": (6.9271, 79.8612)
}


# -----------------------------------------------------------------------------
# Global Header & Disclaimer
# -----------------------------------------------------------------------------
def render_header():
    st.markdown("""
    <div class="hero-banner">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 class="hero-title">🩸 AI-Based Emergency Blood Donor Recommendation System</h1>
                <p class="hero-subtitle">Intelligent Clinical Decision-Support Prototype • Reasoning + Prediction + Recommendation</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Academic & Clinical Decision-Support Notice:</strong>
        This system is an academic AI decision-support prototype operating on synthetic/demo records. It assists authorized healthcare personnel in prioritizing donor contacts during emergencies. 
        <strong>It does not make final clinical eligibility determinations.</strong> Final compatibility (such as physical cross-matching) and donor medical approval must always be conducted by certified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 1: 🏥 DASHBOARD
# -----------------------------------------------------------------------------
def render_dashboard(db: BloodDonorDatabase):
    st.subheader("🏥 Clinical Donor Registry & System Overview")
    
    donors_df = db.get_all_donors()
    requests_df = db.get_all_requests()
    metrics = db.get_summary_metrics()

    # KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['total_donors']:,}</div>
            <div class="metric-label">Total Donors</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #27ae60;">{metrics['available_donors']:,}</div>
            <div class="metric-label">Available Donors</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #2980b9;">{len(ALL_BLOOD_GROUPS)}</div>
            <div class="metric-label">Blood Groups</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #8e44ad;">{metrics['total_requests']}</div>
            <div class="metric-label">Emergency Reqs</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #d35400;">{metrics['avg_response_rate']}%</div>
            <div class="metric-label">Avg Response Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Analytics Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 🩸 Donor Distribution by Blood Group")
        if not donors_df.empty:
            bg_counts = donors_df["blood_group"].value_counts().reindex(ALL_BLOOD_GROUPS, fill_value=0)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            colors = ["#c0392b" if "O" in bg else "#e74c3c" if "A" in bg else "#3498db" if "B" in bg else "#9b59b6" for bg in ALL_BLOOD_GROUPS]
            bars = ax.bar(bg_counts.index, bg_counts.values, color=colors, edgecolor="#ffffff", linewidth=1.5)
            ax.set_ylabel("Registered Donors")
            ax.set_xlabel("Blood Group")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_right:
        st.markdown("##### 📍 Active Availability & City Distribution")
        if not donors_df.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
            
            # Availability Pie
            avail_counts = donors_df["available"].value_counts()
            avail_labels = ["Available (Active)", "Unavailable"]
            avail_colors = ["#2ecc71", "#e74c3c"]
            ax1.pie(avail_counts, labels=avail_labels, autopct="%1.1f%%", startangle=140, colors=avail_colors, textprops={'fontweight': 'bold'})
            ax1.set_title("Current Availability", fontsize=11, fontweight="bold")

            # City Bar Chart
            city_counts = donors_df["city"].value_counts().head(6)
            ax2.barh(city_counts.index[::-1], city_counts.values[::-1], color="#34495e")
            ax2.set_title("Top Donor Hubs", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Donors")
            ax2.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Response Rate & Age Demographics
    st.markdown("<br>", unsafe_allow_html=True)
    c_hist1, c_hist2 = st.columns(2)
    with c_hist1:
        st.markdown("##### 📈 Historical Donor Response Rate Distribution")
        if not donors_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(donors_df["response_rate"] * 100, bins=20, kde=True, color="#2980b9", ax=ax)
            ax.set_xlabel("Historical Response Rate (%)")
            ax.set_ylabel("Donor Count")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c_hist2:
        st.markdown("##### 👥 Donor Age Demographics & Donation History")
        if not donors_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(
                data=donors_df.sample(min(300, len(donors_df)), random_state=42),
                x="age",
                y="donation_count",
                hue="available",
                palette={1: "#27ae60", 0: "#e74c3c"},
                alpha=0.7,
                ax=ax
            )
            ax.set_xlabel("Age (Years)")
            ax.set_ylabel("Total Lifetime Donations")
            ax.legend(title="Available", labels=["Yes (1)", "No (0)"])
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# -----------------------------------------------------------------------------
# PAGE 2: 🚨 EMERGENCY REQUEST INITIATION
# -----------------------------------------------------------------------------
def render_emergency_request_page(db: BloodDonorDatabase, predictor: DonorPredictor):
    st.subheader("🚨 Submit Emergency Blood Requisition")
    st.markdown("Enter emergency recipient parameters. The rule engine, predictive model, and ranking algorithm will prioritize the best active donors instantly.")

    with st.form("emergency_requisition_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            req_blood_group = st.selectbox(
                "Required Blood Group *",
                options=ALL_BLOOD_GROUPS,
                index=1,  # Default O+
                help="The patient's confirmed ABO/Rh blood group."
            )

            hospital_selection = st.selectbox(
                "Receiving Hospital / Medical Center *",
                options=list(PRESET_HOSPITALS.keys()),
                index=0
            )

            # Auto-fill coordinates from hospital selection
            preset_coords = PRESET_HOSPITALS.get(hospital_selection, (6.9271, 79.8612))
            
            coord_c1, coord_c2 = st.columns(2)
            with coord_c1:
                hospital_lat = st.number_input(
                    "Hospital Latitude *",
                    value=float(preset_coords[0]),
                    format="%.4f",
                    help="Synthetic/Demo decimal latitude."
                )
            with coord_c2:
                hospital_lon = st.number_input(
                    "Hospital Longitude *",
                    value=float(preset_coords[1]),
                    format="%.4f",
                    help="Synthetic/Demo decimal longitude."
                )

            units_required = st.number_input(
                "Units Required (Pints) *",
                min_value=1,
                max_value=10,
                value=2,
                step=1
            )

        with col2:
            priority = st.selectbox(
                "Emergency Priority Level *",
                options=["Critical (Immediate OR/ICU)", "High (Within 2 Hours)", "Medium (Within 6 Hours)", "Low (Scheduled Backup)"],
                index=0
            )

            date_c1, date_c2 = st.columns(2)
            with date_c1:
                req_date = st.date_input("Required Date", value=datetime.date.today())
            with date_c2:
                req_time = st.time_input("Required Time", value=datetime.datetime.now().time())

            notes = st.text_area(
                "Clinical Context & Special Requisition Notes",
                placeholder="e.g., Trauma emergency in surgical ICU, cross-matching in progress...",
                height=110
            )

        st.markdown("---")
        st.markdown("##### ⚙️ Pipeline Constraints & Output Settings")
        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            min_days_gap = st.slider("Safe Minimum Gap Since Last Donation (Days)", 30, 180, 90, 10, help="Clinical safe interval guideline (default 90 days).")
        with cfg2:
            top_n = st.selectbox("Number of Top Recommendations to Display", [3, 5, 10], index=1)
        with cfg3:
            enforce_active = st.checkbox("Enforce Active Availability Only", value=True, help="Filter out donors who marked themselves unavailable.")

        submitted = st.form_submit_button("🚨 FIND BEST DONORS", use_container_width=True, type="primary")

    if submitted:
        if not hospital_selection or hospital_lat == 0.0 or hospital_lon == 0.0:
            st.error("Please provide valid hospital details and coordinates.")
            return

        with st.spinner("Executing 3-Pillar AI Pipeline: Deductive Filtering ➔ ML Response Prediction ➔ Multi-Criteria Ranking..."):
            # 1. Fetch registry
            all_donors = db.get_all_donors()
            if all_donors.empty:
                st.error("Donor registry is currently empty. Please generate data first.")
                return

            # Generate Request ID
            req_id = f"REQ-{datetime.datetime.now().strftime('%m%d%H%M%S')}"

            # Save Request into SQLite
            request_record = {
                "request_id": req_id,
                "required_blood_group": req_blood_group,
                "hospital": hospital_selection,
                "latitude": hospital_lat,
                "longitude": hospital_lon,
                "units_required": int(units_required),
                "priority": priority.split()[0],
                "request_date": f"{req_date} {req_time.strftime('%H:%M:%S')}",
                "notes": notes
            }
            db.save_emergency_request(request_record)

            # Stage 1: Rule-Based Reasoning Filter
            rule_engine = RuleEngine(min_days_since_last_donation=min_days_gap, require_availability=enforce_active)
            candidates_df, audit_summary = rule_engine.filter_candidates(all_donors, req_blood_group)

            if candidates_df.empty:
                st.warning(f"⚠️ No currently available compatible donors were found for blood group '{req_blood_group}' with safe gap >= {min_days_gap} days.")
                return

            # Stage 2: Machine Learning Prediction
            predicted_df = predictor.predict_response_probabilities(candidates_df, hospital_lat, hospital_lon)

            # Stage 3: Recommendation Scoring & Ranking
            ranker = DonorRanker()
            ranked_df = ranker.rank_candidates(predicted_df, top_n=top_n)

            # Log Predictions to DB
            pred_logs = []
            for _, row in ranked_df.iterrows():
                pred_logs.append({
                    "request_id": req_id,
                    "donor_id": row["donor_id"],
                    "response_probability": row["response_probability"],
                    "recommendation_score": row["recommendation_score"],
                    "recommendation_label": row["recommendation_label"]
                })
            db.log_predictions(pred_logs)

            # Save to Session State for Recommended Donors Page
            st.session_state["latest_request"] = request_record
            st.session_state["audit_summary"] = audit_summary
            st.session_state["ranked_donors"] = ranked_df
            st.session_state["candidates_count"] = len(predicted_df)

            st.success(f" Analysis complete! Found {len(candidates_df)} eligible candidates. Displaying top {len(ranked_df)} recommendations below.")
            render_recommended_donors_view(request_record, audit_summary, ranked_df)


# -----------------------------------------------------------------------------
# PAGE 3: 🎯 RECOMMENDED DONORS VIEW & EXPLAINABILITY
# -----------------------------------------------------------------------------
def render_recommended_donors_view(request_record: Dict[str, Any], audit_summary: Dict[str, Any], ranked_df: pd.DataFrame):
    st.markdown("---")
    st.subheader(f"🎯 Top Recommended Donors for {request_record['required_blood_group']} ({request_record['hospital']})")

    # Funnel Audit Summary (Demonstrating Deductive Rule-Based Reasoning)
    st.markdown("##### 🔍 Rule-Based Reasoning Filtering Funnel")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        st.markdown(f"""
        <div class="funnel-card">
            <div class="funnel-number">{audit_summary['total_donors']:,}</div>
            <div class="funnel-title">1. Total Donors</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="funnel-card">
            <div class="funnel-number">{audit_summary['compatible_donors']:,}</div>
            <div class="funnel-title">2. Blood-Compatible</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="funnel-card">
            <div class="funnel-number">{audit_summary['available_donors']:,}</div>
            <div class="funnel-title">3. Active Available</div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown(f"""
        <div class="funnel-card">
            <div class="funnel-number">{audit_summary['final_candidates']:,}</div>
            <div class="funnel-title">4. ML Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    with f5:
        st.markdown(f"""
        <div class="funnel-card" style="border: 2px solid #27ae60; background: #eafaf1;">
            <div class="funnel-number" style="color: #27ae60;">{len(ranked_df)}</div>
            <div class="funnel-title" style="color: #27ae60; font-weight: bold;">5. Top Ranked</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"*Compatible Donor Groups for {request_record['required_blood_group']}:* `{', '.join(audit_summary.get('compatible_groups', []))}`")
    st.markdown("<br>", unsafe_allow_html=True)

    # Top Donors Summary Table
    st.markdown("##### 🏆 Ranked Candidates Overview")
    table_display = ranked_df[[
        "rank", "donor_id", "blood_group", "city", "distance_km", "available", 
        "response_probability", "recommendation_score", "recommendation_label"
    ]].copy()
    
    table_display["distance_km"] = table_display["distance_km"].apply(lambda x: f"{x:.1f} km")
    table_display["available"] = table_display["available"].apply(lambda x: "Available" if x == 1 else "Unavailable")
    table_display["response_probability"] = table_display["response_probability"].apply(lambda x: f"{x*100:.1f}%")
    table_display.columns = ["Rank", "Donor ID", "Blood Group", "City", "Distance", "Status", "Response Prob.", "Score (/100)", "Recommendation"]
    
    st.dataframe(table_display, use_container_width=True, hide_index=True)

    # Detailed Explainable Cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🧠 Explainable Recommendation Rationale (XAI)")
    st.markdown("Transparent audit trail detailing why each candidate is prioritized by the AI decision engine.")

    for _, row in ranked_df.iterrows():
        explanation = generate_donor_explanation(
            row, 
            request_record["required_blood_group"],
            hospital_name=request_record["hospital"].split("(")[0].strip()
        )
        
        tier = row["recommendation_label"]
        badge_class = "tier-badge-high" if tier == "Highly Recommended" else "tier-badge-rec" if tier == "Recommended" else "tier-badge-mod" if tier == "Moderately Recommended" else "tier-badge-low"

        with st.expander(f"🏅 Rank #{row['rank']} — Donor {row['donor_id']} | Blood: {row['blood_group']} | Score: {row['recommendation_score']}/100 ({tier})", expanded=(row['rank'] <= 2)):
            col_info, col_breakdown = st.columns([3, 2])
            
            with col_info:
                st.markdown(f"<span class='{badge_class}'>{tier}</span> &nbsp; <strong>Donor ID:</strong> {row['donor_id']} &nbsp; | &nbsp; <strong>City:</strong> {row['city']} &nbsp; | &nbsp; <strong>Age:</strong> {row['age']} yrs", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Evidence-Based Recommendation Rationale:**")
                for bullet in explanation["bullets"]:
                    st.markdown(f"- {bullet}")
                
                # Action Simulation
                st.markdown("<br>", unsafe_allow_html=True)
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button(f"📲 Send Emergency SMS ({row['donor_id']})", key=f"sms_{row['donor_id']}"):
                        st.success(f" Simulated emergency alert dispatched to {row['donor_id']} ({row['blood_group']}) for {request_record['hospital']}!")
                with btn_col2:
                    if st.button(f"📞 Log Direct Call ({row['donor_id']})", key=f"call_{row['donor_id']}"):
                        st.info(f" Call session logged with donor {row['donor_id']}.")

            with col_breakdown:
                st.markdown("**Transparent Scoring Formula Breakdown:**")
                st.caption("Final Score = 40% × P(Response) + 30% × Availability + 20% × Proximity + 10% × History")
                
                breakdown = explanation["component_breakdown"]
                for comp_name, comp_val in breakdown.items():
                    st.write(f"{comp_name}: **{comp_val:.1f} / 100**")
                    st.progress(min(1.0, max(0.0, comp_val / 100.0)))


def render_recommended_donors_page(db: BloodDonorDatabase, predictor: DonorPredictor):
    if "ranked_donors" in st.session_state and "latest_request" in st.session_state:
        render_recommended_donors_view(
            st.session_state["latest_request"],
            st.session_state["audit_summary"],
            st.session_state["ranked_donors"]
        )
    else:
        st.subheader("🎯 Recommended Donors View")
        st.info("No active emergency requisition loaded. Please submit an emergency request or run a demo below.")
        
        if st.button("🚀 Run Standard Demo Request (O+ at Colombo General)"):
            all_donors = db.get_all_donors()
            req_record = {
                "request_id": "REQ-DEMO-001",
                "required_blood_group": "O+",
                "hospital": "National Hospital Colombo",
                "latitude": 6.9271,
                "longitude": 79.8612,
                "units_required": 2,
                "priority": "Critical",
                "request_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notes": "Emergency surgery demo test case"
            }
            rule_engine = RuleEngine(min_days_since_last_donation=90, require_availability=True)
            candidates_df, audit = rule_engine.filter_candidates(all_donors, "O+")
            pred_df = predictor.predict_response_probabilities(candidates_df, 6.9271, 79.8612)
            ranker = DonorRanker()
            ranked_df = ranker.rank_candidates(pred_df, top_n=5)
            
            st.session_state["latest_request"] = req_record
            st.session_state["audit_summary"] = audit
            st.session_state["ranked_donors"] = ranked_df
            st.rerun()


# -----------------------------------------------------------------------------
# PAGE 4: 📋 DONOR DATABASE EXPLORER
# -----------------------------------------------------------------------------
def render_donor_database_page(db: BloodDonorDatabase):
    st.subheader("📋 Searchable Clinical Donor Registry")
    st.markdown("Explore and filter the synthetic donor registry. All records are anonymized demo records for academic decision-support prototyping.")

    donors_df = db.get_all_donors()
    if donors_df.empty:
        st.warning("No donors found in database.")
        return

    # Filter Sidebar / Controls
    with st.expander("🔍 Filter & Search Options", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            sel_bg = st.multiselect("Blood Group", options=ALL_BLOOD_GROUPS, default=ALL_BLOOD_GROUPS)
        with f_col2:
            cities = sorted(donors_df["city"].unique().tolist())
            sel_city = st.multiselect("City / Region", options=cities, default=cities)
        with f_col3:
            sel_avail = st.selectbox("Availability", options=["All Donors", "Available Only", "Unavailable Only"])
        with f_col4:
            min_age, max_age = st.slider("Age Range", 18, 65, (18, 65))

    # Apply filters
    filtered_df = donors_df.copy()
    if sel_bg:
        filtered_df = filtered_df[filtered_df["blood_group"].isin(sel_bg)]
    if sel_city:
        filtered_df = filtered_df[filtered_df["city"].isin(sel_city)]
    if sel_avail == "Available Only":
        filtered_df = filtered_df[filtered_df["available"] == 1]
    elif sel_avail == "Unavailable Only":
        filtered_df = filtered_df[filtered_df["available"] == 0]
    filtered_df = filtered_df[(filtered_df["age"] >= min_age) & (filtered_df["age"] <= max_age)]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(donors_df):,} total donor records**")

    # Format table for clean display
    display_df = filtered_df[[
        "donor_id", "blood_group", "age", "gender", "city", "available", 
        "donation_count", "days_since_last_donation", "previous_requests", 
        "previous_responses", "response_rate", "target_response"
    ]].copy()
    display_df["available"] = display_df["available"].apply(lambda x: "Yes" if x == 1 else "No")
    display_df["response_rate"] = display_df["response_rate"].apply(lambda x: f"{x*100:.0f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # CSV Download
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Filtered Donor Data (CSV)",
        data=csv_data,
        file_name="filtered_donors.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# PAGE 5: 📊 MODEL PERFORMANCE & BENCHMARKING
# -----------------------------------------------------------------------------
def render_model_performance_page():
    st.subheader("📊 Machine Learning Model Benchmarking & Evaluation")
    st.markdown("Comparative performance metrics across classification algorithms trained to predict emergency response likelihood.")

    metrics = get_model_metrics()

    # Comparative Metrics Table
    st.markdown("##### 🏆 Model Benchmarking Matrix")
    model_rows = []
    for model_name in ["Random Forest", "Logistic Regression", "Decision Tree"]:
        if model_name in metrics:
            m = metrics[model_name]
            model_rows.append({
                "Model Algorithm": model_name,
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1 Score": f"{m['f1_score']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "5-Fold CV F1": f"{m.get('cv_f1_mean', 0):.4f} ± {m.get('cv_f1_std', 0):.4f}"
            })
    
    comp_df = pd.DataFrame(model_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.info(f"🌟 **Selected Best Performing Model:** `{metrics.get('best_model_name', 'Random Forest')}` based on optimal F1-score and generalization stability.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrices
    st.markdown("##### 🔲 Confusion Matrix Comparison (Holdout Test Set)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    model_keys = ["Random Forest", "Logistic Regression", "Decision Tree"]
    
    for idx, name in enumerate(model_keys):
        if name in metrics:
            cm = np.array(metrics[name]["confusion_matrix"])
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues" if idx==0 else "Greens" if idx==1 else "Oranges",
                cbar=False, ax=axes[idx],
                xticklabels=["Unlikely (0)", "Likely (1)"],
                yticklabels=["Unlikely (0)", "Likely (1)"]
            )
            axes[idx].set_title(f"{name}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("Actual Label")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Importance for Random Forest
    c_fi, c_report = st.columns([1, 1])

    with c_fi:
        st.markdown("##### 🌲 Random Forest Feature Importance")
        if "feature_importances" in metrics:
            feat_imp = metrics["feature_importances"]
            feat_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            
            fig, ax = plt.subplots(figsize=(7, 5))
            bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color="#c0392b", edgecolor="none")
            ax.set_xlabel("Relative Importance (Gini Impurity Reduction)")
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")
            ax.set_xlim(0, max(feat_df["Importance"]) + 0.05)
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.caption("ℹ️ Feature importance indicates relative statistical weight in the trained model; it represents historical decision factors rather than direct medical causation.")

    with c_report:
        st.markdown("##### 📄 Classification Report (Random Forest)")
        if "Random Forest" in metrics and "classification_report" in metrics["Random Forest"]:
            cr = metrics["Random Forest"]["classification_report"]
            cr_df = pd.DataFrame(cr).transpose()
            st.dataframe(cr_df.style.format("{:.3f}"), use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 6: ℹ️ ABOUT SYSTEM & ARCHITECTURE
# -----------------------------------------------------------------------------
def render_about_page():
    st.subheader("ℹ️ System Architecture & Academic Methodology")
    st.markdown("""
    ### 🩸 AI-Based Emergency Blood Donor Recommendation System
    This system is a clinical decision-support framework engineered to address the critical challenge of rapid, reliable blood donor identification during medical emergencies.
    
    ---
    
    ### 🏛️ The Three-Pillar AI Framework
    The core philosophy bridges **Deductive Clinical Logic**, **Predictive Machine Learning**, and **Multi-Criteria Decision Analysis**:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 1. Rule-Based Reasoning
        * **Deductive Clinical Logic**
        * Validates ABO and Rh red blood cell compatibility matrix.
        * Enforces strict availability and safe donation recovery intervals (≥90 days).
        * Zero ML bypass: Hard clinical rules are non-negotiable.
        """)
    with col2:
        st.markdown("""
        #### 2. Machine Learning
        * **Predictive Classification**
        * Estimates donor affirmative response likelihood $P(\\text{Response} | \\text{Context})$.
        * Incorporates distance, historical response rate, and engagement features.
        * Benchmarked using Random Forest, Logistic Regression, and Decision Trees.
        """)
    with col3:
        st.markdown("""
        #### 3. Intelligent Ranking
        * **Multi-Criteria Optimization**
        * Transparent weighted scoring formula:
          * 40% Response Probability
          * 30% Active Availability
          * 20% Proximity (Haversine km)
          * 10% Donation Track Record
        * Dynamic Explainable AI (XAI) rationale generation.
        """)

    st.markdown("---")
    st.markdown("""
    ### 📐 Mathematical Formulation of Recommendation Scoring
    $$\\text{Recommendation Score} = 0.40 \\cdot S_{\\text{Response}} + 0.30 \\cdot S_{\\text{Avail}} + 0.20 \\cdot S_{\\text{Prox}} + 0.10 \\cdot S_{\\text{Hist}}$$
    
    Where:
    - $S_{\\text{Response}} = P(\\text{Response}) \\times 100$
    - $S_{\\text{Avail}} = \\begin{cases} 100 & \\text{if available} = 1 \\\\ 0 & \\text{otherwise} \\end{cases}$
    - $S_{\\text{Prox}} = \\max(0, 100 - \\text{Distance}_{\\text{km}} \\times 10)$
    - $S_{\\text{Hist}} = \\min(100, \\text{Donation Count} \\times 10)$
    
    ---
    
    ### ⚖️ Ethical Considerations & Transparency
    - **No Fabrication of Medical Fitness**: The system acts strictly as an operational decision-support tool. Final physical testing and clinical cross-matching remain mandatory.
    - **Synthetic Data Usage**: All records are generated programmatically with fixed random seeds for academic reproducibility without handling real patient personal data.
    - **Explainability (XAI)**: Every recommendation is paired with human-readable evidence justifying why the donor was prioritized.
    """)


# -----------------------------------------------------------------------------
# MAIN APPLICATION CONTROLLER
# -----------------------------------------------------------------------------
def main():
    render_header()
    db = get_database()
    predictor = get_predictor()

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/blood-bag.png", width=70)
    st.sidebar.title("Navigation")
    
    nav_option = st.sidebar.radio(
        "Select System Module:",
        [
            "🏥 Dashboard",
            "🚨 Emergency Request",
            "🎯 Recommended Donors",
            "📋 Donor Database",
            "📊 Model Performance",
            "ℹ️ About System"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Quick Actions")
    if st.sidebar.button("🔄 Regenerate Synthetic DB", use_container_width=True):
        save_and_init_db()
        st.sidebar.success("Database re-initialized!")
        st.rerun()

    st.sidebar.markdown("""
    <div style="font-size: 11px; color: #7f8c8d; margin-top: 30px;">
        <strong>AI Blood Donor Recommender</strong><br>
        Version 1.0.0 (Academic Edition)<br>
        Semester 4 AI Group Project
    </div>
    """, unsafe_allow_html=True)

    # Route navigation
    if nav_option == "🏥 Dashboard":
        render_dashboard(db)
    elif nav_option == "🚨 Emergency Request":
        render_emergency_request_page(db, predictor)
    elif nav_option == "🎯 Recommended Donors":
        render_recommended_donors_page(db, predictor)
    elif nav_option == "📋 Donor Database":
        render_donor_database_page(db)
    elif nav_option == "📊 Model Performance":
        render_model_performance_page()
    elif nav_option == "ℹ️ About System":
        render_about_page()


if __name__ == "__main__":
    main()

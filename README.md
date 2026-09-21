# 🩸 AI-Based Emergency Blood Donor Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg)](https://www.sqlite.org/)
[![Status](https://img.shields.io/badge/Academic-Prototype-success.svg)](#)

> **Clinical & Academic Disclaimer**:  
> This system is an **academic AI decision-support prototype** operating on reproducible synthetic/demo data. It assists authorized medical personnel in prioritizing contacts during emergencies. **It does NOT make final clinical eligibility determinations.** Final compatibility (e.g., cross-matching) and donor physical eligibility must always be verified by certified healthcare professionals.

---

## 1. Project Description
The **AI-Based Emergency Blood Donor Recommendation System** is an intelligent, transparent clinical decision-support application. During acute emergency blood shortages (e.g., trauma surgeries, postpartum hemorrhages, major surgical complications), medical coordinators must quickly identify, filter, and contact suitable blood donors. This system bridges deterministic clinical rules, machine learning response prediction, and multi-criteria ranking to provide explainable top-N donor recommendations in real-time.

---

## 2. Problem Statement
In emergency medicine, conventional blood bank registries often face severe operational bottlenecks:
- Contacting donors sequentially from unranked static lists leads to critical delays.
- Many contacted donors are geographically too distant or clinically ineligible due to recent donations.
- Donors may be marked active but have low historical responsiveness during emergencies.
- Traditional systems lack explainability, leaving coordinators uncertain about why a specific donor was prioritized.

---

## 3. Objectives
1. **Deductive Filtering**: Enforce non-negotiable medical rules (ABO & Rh red blood cell compatibility, active availability, safe recovery intervals).
2. **Predictive Analytics**: Estimate the likelihood that an eligible donor will affirmatively respond to an emergency requisition using Machine Learning.
3. **Multi-Criteria Optimization**: Rank candidates transparently based on response probability, proximity, active status, and historical engagement.
4. **Explainable AI (XAI)**: Provide human-readable, evidence-based rationales for every recommended candidate.
5. **Interactive Healthcare UI**: Deliver a responsive Streamlit dashboard for emergency requisitioning, registry exploration, and ML benchmarking.

---

## 4. The Three-Pillar AI Architecture

```
+-----------------------------------------------------------------------------------+
|                            EMERGENCY BLOOD REQUISITION                            |
| (Blood Group, Hospital Coordinates, Units Required, Urgency Priority Level, Date) |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                   STAGE 1: RULE-BASED REASONING (Deductive AI)                    |
|  - ABO & Rh Red Blood Cell Compatibility Matrix verification                      |
|  - Active Availability Verification (available == 1)                              |
|  - Safe Donation Gap Constraints (days_since_last_donation >= 90 days)            |
|  - Generates transparent multi-stage filtering audit funnel                       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|               STAGE 2: MACHINE LEARNING CLASSIFICATION (Predictive AI)             |
|  - Feature Engineering (Haversine distance to hospital, response rate history)    |
|  - Benchmarked Models (Random Forest, Logistic Regression, Decision Tree)         |
|  - Predicts Response Probability: P(Affirmative Response | Emergency Context)     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|            STAGE 3: MULTI-CRITERIA RECOMMENDATION & RANKING (Decision AI)         |
|  - Transparent Weighted Scoring: 40% ML Prob + 30% Avail + 20% Prox + 10% History|
|  - Dynamic Evidence-Based Explainable AI (XAI) rationale generation               |
|  - Stratified Recommendation Tiers (Highly Recommended, Recommended, etc.)        |
|  - Top-N Selection and Emergency Alert Dispatch Simulation                        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                        STREAMLIT HEALTHCARE DASHBOARD & UI                        |
|  - Analytics Dashboard | Emergency Request | Recommended Donors | DB | Evaluation |
+-----------------------------------------------------------------------------------+
```

---

## 5. Dataset Description
The dataset contains **1,400 synthetic donor records** generated with realistic probabilistic relationships using a fixed seed (`seed=42`):

| Column | Type | Description |
|---|---|---|
| `donor_id` | String | Unique identifier (`D0001` to `D1400`) |
| `blood_group` | String | ABO/Rh group (`O+`, `O-`, `A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`) |
| `age` | Integer | Donor age (18 to 65 years) |
| `gender` | String | Gender category (`Male`, `Female`) |
| `city` | String | Synthetic regional hub (e.g., Colombo, Kandy, Galle, Gampaha) |
| `latitude` | Float | Geographic latitude coordinate |
| `longitude` | Float | Geographic longitude coordinate |
| `available` | Integer | Active availability status (1 = Available, 0 = Unavailable) |
| `donation_count` | Integer | Total lifetime donations (1 to 25) |
| `days_since_last_donation` | Integer | Days elapsed since previous donation (30 to 450) |
| `previous_requests` | Integer | Total past emergency notifications sent to donor |
| `previous_responses` | Integer | Number of affirmative responses by donor |
| `response_rate` | Float | Historical response ratio (`previous_responses / previous_requests`) |
| `contacted_before` | Integer | Whether previously contacted (1 = Yes, 0 = No) |
| `target_response` | Integer | ML Classification Target (1 = Likely Response, 0 = Unlikely) |

---

## 6. Blood Group Compatibility Rules

The rule engine implements standard red blood cell (RBC) compatibility:

| Recipient Blood Group | Compatible Donor Blood Groups |
|---|---|
| **O-** | `O-` |
| **O+** | `O-`, `O+` |
| **A-** | `O-`, `A-` |
| **A+** | `O-`, `O+`, `A-`, `A+` |
| **B-** | `O-`, `B-` |
| **B+** | `O-`, `O+`, `B-`, `B+` |
| **AB-** | `O-`, `A-`, `B-`, `AB-` |
| **AB+** | `O-`, `O+`, `A-`, `A+`, `B-`, `B+`, `AB-`, `AB+` *(Universal Recipient)* |

---

## 7. Machine Learning Methodology

### Algorithms Evaluated:
1. **Random Forest Classifier** (`n_estimators=120`, `max_depth=8`) — *Selected Best Model*
2. **Logistic Regression** (Standardized pipeline, L2 regularization)
3. **Decision Tree Classifier** (`max_depth=6`, `min_samples_split=6`)

### Benchmarking Results (Stratified 80/20 Holdout & 5-Fold Cross-Validation):
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 |
|---|---|---|---|---|---|---|
| **Random Forest** | **91.07%** | **88.82%** | **96.18%** | **92.35%** | **0.9486** | **0.9182 ± 0.012** |
| **Logistic Regression** | 89.64% | 88.10% | 94.27% | 91.08% | 0.9273 | 0.9065 ± 0.014 |
| **Decision Tree** | 91.07% | 88.82% | 96.18% | 92.35% | 0.9377 | 0.9080 ± 0.018 |

---

## 8. Multi-Criteria Recommendation Algorithm

$$\text{Final Recommendation Score} = 0.40 \times S_{\text{Response}} + 0.30 \times S_{\text{Avail}} + 0.20 \times S_{\text{Prox}} + 0.10 \times S_{\text{Hist}}$$

### Component Sub-Scores (Normalized 0–100):
- **Response Probability Score ($S_{\text{Response}}$)**: $P(\text{Response}) \times 100$
- **Availability Score ($S_{\text{Avail}}$)**: $100$ if active, else $0$
- **Proximity Score ($S_{\text{Prox}}$)**: $\max(0, 100 - \text{distance}_{\text{km}} \times 10)$
- **Donation History Score ($S_{\text{Hist}}$)**: $\min(100, \text{donation\_count} \times 10)$

### Stratified Recommendation Tiers:
- **90 – 100**: `Highly Recommended` 🟢
- **75 – 89**: `Recommended` 🔵
- **60 – 74**: `Moderately Recommended` 🟡
- **< 60**: `Low Priority` 🔴

---

## 9. Project Structure

```
blood-donor-ai/
│
├── data/
│   ├── donors.csv                     # Synthetic donor registry (1,400 records)
│   ├── emergency_requests.csv         # Emergency requisition logs
│   └── blood_donor.db                 # SQLite relational database
│
├── models/
│   ├── donor_response_model.pkl       # Serialized Random Forest model
│   └── model_metrics.pkl              # Evaluation benchmark package
│
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── blood_compatibility.py         # Rule-based ABO/Rh compatibility engine
│   ├── data_generator.py              # Reproducible dataset generator & DB init
│   ├── database.py                    # SQLite database interface
│   ├── preprocessing.py               # Haversine distance & feature transformers
│   ├── rule_engine.py                 # Multi-stage deterministic filter
│   ├── train_model.py                 # ML training & comparative benchmarking
│   ├── prediction.py                  # ML inference & response probability
│   └── ranking.py                     # Multi-criteria scoring & XAI engine
│
├── notebooks/
│   └── model_training.ipynb           # University-level EDA & training notebook
│
├── verify_pipeline.py                 # Automated end-to-end test suite
├── app.py                             # Interactive Streamlit application
├── requirements.txt                   # Python package dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # Comprehensive documentation
```

---

## 10. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Pip package manager

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd blood-donor-ai
```

### Step 2: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

---

## 11. Execution Guide

### 1. Generate Synthetic Dataset & Initialize Database
```bash
python src/data_generator.py
```

### 2. Train & Benchmark ML Models
```bash
python src/train_model.py
```

### 3. Run Pipeline Verification Test
```bash
python verify_pipeline.py
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 12. Example Test Workflow

**Scenario**: Emergency trauma surgery requiring **O+** blood at **National Hospital Colombo** (Coordinates: `6.9271, 79.8612`).

```
================================================================================
                    TOP RECOMMENDED DONORS FOR O+ AT COLOMBO                    
================================================================================
Rank  | Donor ID  | Blood  | Distance   | Avail  | Prob     | Score  | Tier
--------------------------------------------------------------------------------
1     | D0149     | O+     | 1.27 km    | 1      | 96.8%   | 95.2   | Highly Recommended
2     | D0083     | O+     | 2.11 km    | 1      | 98.2%   | 95.1   | Highly Recommended
3     | D0791     | O+     | 2.06 km    | 1      | 97.3%   | 94.8   | Highly Recommended
4     | D0608     | O+     | 1.14 km    | 1      | 97.0%   | 94.5   | Highly Recommended
5     | D0899     | O+     | 2.59 km    | 1      | 99.0%   | 94.4   | Highly Recommended

--------------------------------------------------------------------------------
EXPLANATION FOR TOP DONOR (D0149):
  ✓ Identical match (O+ to O+)
  ✓ Currently active and marked available for emergency dispatch
  ✓ Exceptional proximity: only 1.27 km from National Hospital Colombo
  ✓ 96.8% ML-predicted probability of affirmative emergency response
  ✓ Experienced donor (9 prior donations; 409 days since last donation)
--------------------------------------------------------------------------------
```

---

## 13. System Limitations
- **Synthetic Geographic Modeling**: Proximity is based on Euclidean/Haversine spherical distance rather than live Google Maps/OSM road network traffic routing.
- **Simplified Serology**: Real clinical blood transfusion involves minor antigen sub-typing (Kell, Duffy, Kidd) and laboratory cross-matching not captured in high-level ABO/Rh systems.
- **Demo Behavioral Data**: Real-world response rates vary based on communication channels (SMS vs automated voice call vs mobile app push notifications).

---

## 14. Ethical Considerations & Privacy
- **Privacy by Design**: No real patient or donor personal identifying information (PII) is included or stored.
- **Non-Autonomous Medical Decisions**: The system acts strictly as an operational assistant. Human medical coordinators make all contact and allocation decisions.
- **Fairness & Non-Discrimination**: The ranking algorithm uses geographic and engagement metrics without demographic penalization.

---

## 15. Future Improvements
- [ ] Integration with real-time GPS fleet tracking and live traffic APIs (e.g., OSRM, Google Maps).
- [ ] Multi-hospital regional inventory exchange optimization.
- [ ] Automated SMS/WhatsApp API gateway integration for one-click donor dispatch.
- [ ] Deep reinforcement learning for dynamic threshold tuning during mass casualty triage.

---

## 16. Academic Attribution
* Developed as part of the **Semester 4 AI Group Project** (AI Decision Support in Healthcare).
* Decision Support Prototype • Deductive Reasoning + Predictive Machine Learning + Multi-Criteria Optimization.

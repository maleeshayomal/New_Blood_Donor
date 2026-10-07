# Data Dictionary: AI Emergency Blood Donor Recommendation System

## 1. Document Purpose & Synthetic Data Statement

### Purpose
This document provides a comprehensive technical reference for all data assets utilized in the AI-Based Emergency Blood Donor Recommendation System. It describes the schema, data types, statistical distributions, programmatic generation logic, target classification labels, and database tables.

### Synthetic Data Declaration
> **Notice:** All data within this project is **100% synthetic**. It was generated programmatically by [`src/data_generator.py`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/src/data_generator.py) using a fixed random seed (`seed=42`) with NumPy and Python standard libraries. No real donor, patient, or clinical personal health information (PHI) is included.

---

## 2. Donor Dataset Schema (`data/donors.csv`)

The primary dataset file is [`data/donors.csv`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/data/donors.csv). It contains **1,400 rows** and **15 columns**, with no missing (`NaN`) values.

### Summary Table of All Columns

| Column Name | Data Type | Observed Range / Categories | Mean / Distribution | Generation Logic in `src/data_generator.py` |
|---|---|---|---|---|
| `donor_id` | String (`object`) | `D0001` to `D1400` | 1,400 unique values | Formatted sequential string: `f"D{i+1:04d}"` for `i` in `0..1399`. |
| `blood_group` | String (`object`) | 8 categories: `O+`, `A+`, `B+`, `O-`, `A-`, `AB+`, `B-`, `AB-` | `O+`: 547 (39.07%)<br>`A+`: 390 (27.86%)<br>`B+`: 211 (15.07%)<br>`O-`: 83 (5.93%)<br>`A-`: 66 (4.71%)<br>`AB+`: 50 (3.57%)<br>`B-`: 42 (3.00%)<br>`AB-`: 11 (0.79%) | Sampled using `np.random.choice` with empirical population weights: `[0.38, 0.07, 0.28, 0.05, 0.15, 0.03, 0.03, 0.01]`. |
| `age` | Integer (`int64`) | Min: 18<br>Max: 64 | Mean: 32.69 years | Normal distribution $\mathcal{N}(32, 10^2)$ rounded to integer and clipped to the legal donation range $[18, 65]$: `np.clip(np.round(np.random.normal(32, 10)), 18, 65)`. |
| `gender` | String (`object`) | 2 categories: `Male`, `Female` | `Male`: 740 (52.86%)<br>`Female`: 660 (47.14%) | Sampled with `np.random.choice(["Male", "Female"], p=[0.52, 0.48])`. |
| `city` | String (`object`) | 10 regional hubs: Colombo, Kandy, Gampaha, Negombo, Kalutara, Galle, Kurunegala, Matara, Ratnapura, Anuradhapura | Colombo: 420 (30.0%)<br>Kandy: 231 (16.5%)<br>Gampaha: 190 (13.6%)<br>Negombo: 112 (8.0%)<br>Kalutara: 110 (7.9%)<br>Galle: 105 (7.5%)<br>Kurunegala: 83 (5.9%)<br>Matara: 54 (3.9%)<br>Ratnapura: 49 (3.5%)<br>Anuradhapura: 46 (3.3%) | Sampled from regional hub list with weights: `[0.30, 0.15, 0.15, 0.08, 0.08, 0.08, 0.06, 0.04, 0.03, 0.03]`. |
| `latitude` | Float (`float64`) | Min: 5.9156<br>Max: 8.3450 | Mean: 6.9693 | Base city latitude plus Gaussian jitter: `round(base_lat + np.random.normal(0, 0.02), 4)`. |
| `longitude` | Float (`float64`) | Min: 79.7778<br>Max: 80.6837 | Mean: 80.1338 | Base city longitude plus Gaussian jitter: `round(base_lon + np.random.normal(0, 0.02), 4)`. |
| `available` | Integer (`int64`) | Binary: `0` or `1` | `1` (Available): 1,013 (72.36%)<br>`0` (Unavailable): 387 (27.64%)<br>Mean: 0.7236 | Sampled with `np.random.choice([1, 0], p=[0.72, 0.28])`. |
| `donation_count` | Integer (`int64`) | Min: 1<br>Max: 25 | Mean: 4.10 donations | Geometric distribution clipped to $[1, 25]$: `np.clip(np.random.geometric(p=0.25), 1, 25)`. |
| `days_since_last_donation` | Integer (`int64`) | Min: 30<br>Max: 449 | Mean: 237.18 days | Uniform integer sampled between 30 and 449: `np.random.randint(30, 450)`. |
| `previous_requests` | Integer (`int64`) | Min: 2<br>Max: 32 | Mean: 11.61 requests | Sum of `donation_count` and a uniform random integer between 1 and 14, clipped to $[1, 35]$: `np.clip(donation_counts + np.random.randint(1, 15), 1, 35)`. |
| `previous_responses` | Integer (`int64`) | Min: 0<br>Max: 25 | Mean: 4.54 responses | Calculated as `round(raw_response_prop * previous_requests)`, where `raw_response_prop` is drawn from a Beta distribution scaled by donation experience: `np.clip(np.random.beta(3.5, 2.0) * (0.5 + 0.5 * (donation_counts / 25.0)), 0.05, 0.98)`. |
| `response_rate` | Float (`float64`) | Min: 0.00<br>Max: 0.86 | Mean: 0.3778 (37.78%) | Exact ratio rounded to 2 decimal places: `np.round(previous_responses / previous_requests, 2)`. |
| `contacted_before` | Integer (`int64`) | Value: `1` (Constant) | 1,400 rows with value `1` (100%) | Binary flag: `np.where(previous_requests > 0, 1, 0)`. |
| `target_response` | Integer (`int64`) | Binary: `0` or `1` | `1` (Likely): 786 (56.14%)<br>`0` (Unlikely): 614 (43.86%)<br>Mean: 0.5614 | Generated via probabilistic log-odds formula with logistic sigmoid and an availability penalty. |

---

## 3. Production of the Target Column (`target_response`)

The binary classification target `target_response` indicates whether a registered donor is predicted to agree affirmatively to an emergency requisition ($1 = \text{Likely}$, $0 = \text{Unlikely}$).

### Exact Code Formula in `src/data_generator.py`

Lines 130–149 of [`src/data_generator.py`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/src/data_generator.py#L130-L149) define how `target_response` is synthesized:

```python
log_odds = (
    -2.0
    + (2.2 * available)
    + (3.0 * (response_rate - 0.5))
    + (0.06 * donation_counts)
    + np.where(days_since_last >= 90, 0.6, -1.5)
    + (0.4 * contacted_before)
    - (0.015 * (ages - 30))
    + np.random.normal(0, 0.75, size=num_records)  # Realistic stochastic noise
)

# Sigmoid function for probability
probs = 1.0 / (1.0 + np.exp(-log_odds))
target_response = (probs >= 0.50).astype(int)

# Unavailable donors almost never respond (physical constraint override with 96% penalty)
for i in range(num_records):
    if available[i] == 0 and np.random.rand() < 0.96:
        target_response[i] = 0
```

### Explanation of Variables Influencing `target_response`

1. **`available` (Heavy Positive Influence + Hard Override):**
   - Adds $+2.2$ to the log-odds calculation when `available == 1`.
   - Furthermore, a post-processing rule enforces that if `available == 0`, `target_response` is overridden to `0` with a 96% probability.
2. **`response_rate` (Strong Positive Influence):**
   - Adds $+3.0 \times (\text{response\_rate} - 0.5)$ to log-odds. Donors with historical response rates above 50% receive a positive boost; those below 50% receive a penalty.
3. **`days_since_last_donation` (Thresholded Step-Function):**
   - If $\ge 90$ days (clinically safe interval), adds $+0.6$.
   - If $< 90$ days (insufficient recovery gap), penalizes with $-1.5$.
4. **`donation_count` (Moderate Positive Influence):**
   - Adds $+0.06$ per lifetime donation, reflecting increased reliability among experienced donors.
5. **`ages` (Mild Negative Influence):**
   - Subtracts $0.015 \times (\text{age} - 30)$, slightly reducing likelihood as donor age increases beyond 30.
6. **`contacted_before` (Nominal Constant Term):**
   - Adds $+0.4$ to log-odds. Because `contacted_before` is identically 1 for all rows, it acts effectively as a fixed intercept term.
7. **Stochastic Noise:**
   - Gaussian noise $\mathcal{N}(0, 0.75^2)$ ensures probabilistic realism and avoids deterministic separability.

### Variables NOT Part of the `target_response` Formula

- **`distance_km` is NOT part of the `target_response` generation in `src/data_generator.py`:**
  - `distance_km` does not exist in `data/donors.csv`.
  - When training models in [`src/train_model.py`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/src/train_model.py), `distance_km` is computed dynamically relative to a reference coordinate (Colombo General Hospital). In live inference (`src/prediction.py`), `distance_km` is computed dynamically from the user-selected hospital coordinates.
- **`blood_group`**: Not included in the formula (compatibility is handled upstream by the rule engine).
- **`gender`**: Not included in the formula.
- **`city`, `latitude`, `longitude`**: Not included in the formula.
- **`previous_requests` and `previous_responses`**: Not directly added; they influence log-odds exclusively through `response_rate`.

---

## 4. Class Balance Analysis

Class distribution was computed directly from [`data/donors.csv`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/data/donors.csv):

```
Total Donors in Dataset: 1,400
```

### A. Across All Donors (Entire Registry)
- `target_response = 1` (Likely to respond): **786 records (56.14%)**
- `target_response = 0` (Unlikely to respond): **614 records (43.86%)**

### B. Across Rule-Eligible Donors (`available == 1` AND `days_since_last_donation >= 90`)
In actual emergency requisitions, the rule engine filters out unavailable donors and donors with $< 90$ days recovery gap before passing candidates to the model.
- Total rule-eligible donors: **864 records (61.71% of total registry)**
- `target_response = 1` among rule-eligible donors: **768 records (88.89%)**
- `target_response = 0` among rule-eligible donors: **96 records (11.11%)**

*Observation:* Because active availability and safe donation intervals are strongly correlated with positive response log-odds in data generation, the candidate pool entering stage 2 has a high positive response density (~88.9%).

---

## 5. Constant and Model-Unused Columns

### 1. Constant Feature: `contacted_before`
- **Observed Values:** Constant integer `1` for all 1,400 records ($\min = 1, \max = 1, \text{mean} = 1.0000, \text{variance} = 0$).
- **Cause:** In [`src/data_generator.py`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/src/data_generator.py#L102-L105), `previous_requests` is generated as `np.clip(donation_counts + np.random.randint(1, 15), 1, 35)`. Because `donation_counts >= 1` and `randint(1, 15) >= 1`, `previous_requests` is always $\ge 2$. Consequently, `contacted_before = np.where(previous_requests > 0, 1, 0)` is universally true (`1`).
- **Impact on ML Model:** Although included in `ML_FEATURE_NAMES`, its feature importance in the trained Random Forest model is verified to be **`0.0000`** because zero variance provides no decision splitting power.

### 2. Registry Columns Not Passed to the ML Model
The following columns from `data/donors.csv` are not in `ML_FEATURE_NAMES` and are not fed directly into scikit-learn models:
- `donor_id`: Unique primary key string identifier (used for presentation and SQL joins).
- `blood_group`: Categorical string used by `RuleEngine` for ABO/Rh matrix filtering, but omitted from ML inference feature vector.
- `gender`: Demographic descriptor; omitted to prevent demographic bias in recommendation ranking.
- `city`: Geographic location name; used for display filtering in Streamlit.
- `latitude` and `longitude`: Raw coordinates used to compute `distance_km` via the Haversine formula; raw coordinates themselves are not input features.

---

## 6. SQLite Relational Database Schema (`data/blood_donor.db`)

The SQLite database file is located at [`data/blood_donor.db`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/data/blood_donor.db). It contains 4 tables defined and managed by [`src/database.py`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/src/database.py) and [`src/data_generator.py`](file:///c:/Users/USER/Documents/KDU/SEM4/group_project/New_Blood_Donor/src/data_generator.py).

### Database Tables Overview

| Table Name | Purpose | Populated By | Read By | Verified Current Row Count |
|---|---|---|---|---|
| `donors` | Primary donor registry storing all baseline profiles, geographic coordinates, and historical donation metrics. | `src/data_generator.py:save_and_init_db()` | `src/database.py:get_all_donors()` | **1,400 rows** |
| `emergency_requests` | Logs submitted emergency blood requisitions, including required blood group, hospital coordinates, units, priority, and date. | `src/data_generator.py:save_and_init_db()` (seed data)<br>`src/database.py:save_emergency_request()` (form submissions) | `src/database.py:get_all_requests()` | **8 rows** |
| `predictions` | Logs top-ranked donor recommendations and predicted probabilities for each processed emergency requisition. | `src/database.py:log_predictions()` | SQLite queries (audit logging) | **30 rows** |
| `donation_history` | Intended for longitudinal records of past individual donation transactions. | *None (Unused)* | *None (Unused)* | **0 rows (Empty)** |

### Detailed Table Schemas

#### 1. Table: `donors`
- `donor_id` (`TEXT PRIMARY KEY`): Unique donor identifier (e.g., `D0001`).
- `blood_group` (`TEXT NOT NULL`): ABO/Rh group (`O+`, `O-`, `A+`, etc.).
- `age` (`INTEGER`): Age in years.
- `gender` (`TEXT`): Gender (`Male`, `Female`).
- `city` (`TEXT`): Synthetic city of residence.
- `latitude` (`REAL`): Latitude coordinate.
- `longitude` (`REAL`): Longitude coordinate.
- `available` (`INTEGER`): Availability flag (`1` = Available, `0` = Unavailable).
- `donation_count` (`INTEGER`): Lifetime donation count.
- `days_since_last_donation` (`INTEGER`): Days since most recent donation.
- `previous_requests` (`INTEGER`): Number of past notifications received.
- `previous_responses` (`INTEGER`): Number of affirmative responses.
- `response_rate` (`REAL`): Historical response ratio (`0.00` to `1.00`).
- `contacted_before` (`INTEGER`): Prior contact flag (`1` or `0`).
- `target_response` (`INTEGER`): Synthetic ML target (`1` or `0`).

#### 2. Table: `emergency_requests`
- `request_id` (`TEXT PRIMARY KEY`): Unique requisition code (e.g., `REQ-1001` or `REQ-MMDDHHMMSS`).
- `required_blood_group` (`TEXT NOT NULL`): Blood group required for the patient.
- `hospital` (`TEXT NOT NULL`): Name of requesting medical center.
- `latitude` (`REAL`): Destination hospital latitude.
- `longitude` (`REAL`): Destination hospital longitude.
- `units_required` (`INTEGER`): Units of blood needed.
- `priority` (`TEXT`): Urgency tier (`Critical`, `High`, `Medium`, `Low`).
- `request_date` (`TEXT`): Timestamp of the requisition.
- `notes` (`TEXT`): Clinical notes or context.

#### 3. Table: `predictions`
- `prediction_id` (`INTEGER PRIMARY KEY AUTOINCREMENT`): Auto-incrementing identifier.
- `request_id` (`TEXT`): Associated requisition ID (`FOREIGN KEY -> emergency_requests`).
- `donor_id` (`TEXT`): Associated donor ID (`FOREIGN KEY -> donors`).
- `response_probability` (`REAL`): ML predicted response probability ($0.0$ to $1.0$).
- `recommendation_score` (`REAL`): Multi-criteria score ($0.0$ to $100.0$).
- `recommendation_label` (`TEXT`): Tier label (e.g., `Highly Recommended`).
- `created_at` (`TIMESTAMP DEFAULT CURRENT_TIMESTAMP`): Record creation timestamp.

#### 4. Table: `donation_history` (Unused)
- `donation_id` (`INTEGER PRIMARY KEY AUTOINCREMENT`)
- `donor_id` (`TEXT`, `FOREIGN KEY -> donors`)
- `donation_date` (`TEXT`)
- `response` (`INTEGER`)
- `request_id` (`TEXT`)
- *Status Note:* This table is created in the database schema (`src/database.py:_init_tables` and `src/data_generator.py:save_and_init_db`) for potential future transaction logging, but currently has **0 rows** and is not written to or read by any application module.

# CrowdCast AI — Smart Public Transport Crowd Forecasting System

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)
![Interface](https://img.shields.io/badge/App-Streamlit-red)
![Visuals](https://img.shields.io/badge/Charts-Plotly-teal)
![Testing](https://img.shields.io/badge/Tests-pytest-green)

> **NOTICE — SYNTHETIC DEMONSTRATION DATA:**
> The dataset included with this project is synthetically generated for demonstration, modeling validation, and academic defense. It exhibits realistic commuter patterns (morning/evening rush hour peaks, weekend leisure shifts, adverse weather ridership surges, school term demand, and event-driven spikes). It does not represent actual public transport authority telemetry and is not intended for life-critical transit scheduling. It can be directly replaced with real Automatic Passenger Counter (APC) or smart-card ticketing logs.

---

## 1. Project Overview

**CrowdCast AI** is a machine learning-powered decision support system designed to predict commuter crowd levels and forecast exact passenger loads across urban public transit networks. By integrating route topology, calendar schedules, temporal departure hours, meteorological conditions, and campus/special event triggers, the system helps both:
1. **Transit Dispatchers**: Dynamically manage vehicle frequency, deploy auxiliary relief buses, and avoid dangerous crush loads.
2. **Daily Commuters**: Make informed departure decisions, anticipate crowding, and choose less congested travel windows.

---

## 2. Technology Stack

- **Python 3.10+**: Core programming language.
- **pandas & NumPy**: Vectorized data manipulation and feature engineering.
- **scikit-learn**: Preprocessing pipelines (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`) and dual tree ensemble models (`RandomForestClassifier`, `RandomForestRegressor`).
- **Streamlit**: Responsive, interactive web dashboard styled with a modern Navy/Teal aesthetic.
- **Plotly**: Interactive charts, probability distributions, hourly demand curves, and confusion matrix heatmaps.
- **pytest**: Automated unit and regression test suite verifying threshold boundaries, data schemas, and error tolerance.

---

## 3. Core Machine Learning Formulation

CrowdCast AI employs a **dual-pipeline ensemble architecture**:

```
Input Features (Route, Day, Hour, Weather, Event, Term, Duration)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│     Categorical Pipeline      │   │      Numerical Pipeline       │
│  - Mode SimpleImputer         │   │  - Median SimpleImputer       │
│  - OneHotEncoder              │   │  - StandardScaler             │
│    (handle_unknown='ignore')  │   │                               │
└───────────────┬───────────────┘   └───────────────┬───────────────┘
                └───────────────────┬───────────────┘
                                    ▼
                          ColumnTransformer (X)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│     RandomForestClassifier      │   │      RandomForestRegressor      │
│  - Predicts Crowd Level:        │   │  - Estimates Passenger Count    │
│    (Low / Moderate / High)      │   │  - Inter-tree variance (± std)  │
│  - Outputs Class Probabilities  │   │                                 │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

### Features Utilized:
1. `route_id`: Categorical identifier (e.g., Downtown Express, Uptown Metro, Suburban Feeder, Campus Line, Crosstown, Airport Flyer).
2. `day_type`: Categorical (`Weekday`, `Weekend`, `Holiday`).
3. `departure_hour`: Numerical departure hour (5:00 AM to 11:00 PM / 05:00–23:00).
4. `weather`: Categorical (`Clear`, `Rainy`, `Overcast`, `Stormy`, `Snow/Fog`).
5. `nearby_event`: Binary flag ($1$ = concert/game/expo near transit stops, $0$ = normal).
6. `college_term`: Binary flag ($1$ = semester in session, $0$ = academic break).
7. `trip_duration`: Scheduled route trip duration in minutes (12 to 85 minutes).

### Crowd-Level Thresholds:
| Crowd Level | Passenger Count Range | Operational Interpretation |
| :--- | :--- | :--- |
| **Low** | $< 35$ passengers | Ample seating available; relaxed travel. |
| **Moderate** | $35 - 70$ passengers | Most seats occupied; light standing room available. |
| **High** | $> 70$ passengers | Packed capacity; heavy standing; potential boarding delays. |

---

## 4. Directory Structure

```
c:\Users\user\Desktop\ML\
├── app.py                         # Streamlit interactive application
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Comprehensive project documentation
├── data/
│   └── crowdcast_sample_data.csv  # Synthetic demonstration dataset (2,000 rows)
├── src/
│   ├── __init__.py                # Package initialization
│   ├── preprocessing.py          # Feature definitions, pipelines, threshold logic
│   ├── data_generator.py          # Synthetic dataset generation & serialization
│   └── model.py                   # Model training, evaluation, inference functions
└── tests/
    ├── __init__.py                # Test package initialization
    ├── test_data.py               # Unit tests for thresholds and data schemas
    └── test_models.py             # Unit tests for ML models, metrics, and error handling
```

---

## 5. Quickstart & Installation Guide (Windows PowerShell)

Follow these exact commands in Windows PowerShell:

### Step 1: Open PowerShell and Navigate to the Project Root
```powershell
cd c:\Users\user\Desktop\ML
```

### Step 2: Create and Activate Virtual Environment
```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```powershell
pytest -v
```

### Step 5: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 6. Model Evaluation Highlights

Evaluation is performed on a **held-out 20% test partition (N = 400 trips)** isolated prior to model fitting:

- **Classification Accuracy**: $\approx 88 - 93\%$ on held-out test split.
- **Classification Metrics**: Balanced per-class Precision, Recall, and F1 across Low, Moderate, and High classes.
- **Regression MAE**: $\approx 3.2 - 4.5$ passengers mean absolute error.
- **Regression $R^2$ Score**: $\approx 0.92 - 0.96$, demonstrating that variance in ridership is reliably accounted for by the temporal and environmental features.

---

## 7. Viva & Oral Defense Preparation

### Q1: Why use both a classifier and a regressor instead of just one?
**Answer**: Public transit dispatchers and riders have different operational needs. Passengers benefit from a quick, intuitive categorical classification ("Low / Moderate / High") with probability confidence. Conversely, transit planners need continuous passenger count estimates to calculate fleet load factors, fuel utilization, and bus capacity percentages. Training both models provides both levels of granularity while allowing the regressor's ensemble variance ($\sigma_{\text{trees}}$) to serve as an uncertainty metric.

### Q2: How is data leakage prevented during preprocessing?
**Answer**: Feature transformations (median imputation and standard scaling for numerical features; mode imputation and one-hot encoding for categorical features) are strictly wrapped inside a scikit-learn `Pipeline` and fit **only** on the training split ($X_{\text{train}}$). Held-out test data ($X_{\text{test}}$) and new inference inputs are only transformed, never fit.

### Q3: How does the application handle unseen categories in production?
**Answer**: The categorical pipeline sets `OneHotEncoder(handle_unknown='ignore')`. If a user inputs an unexpected transit route (e.g., `Route_999`) or rare weather condition, the encoder produces all zeros for those category columns without throwing an error, allowing the tree ensemble to generate sensible predictions from the remaining features.

### Q4: How would you replace the synthetic dataset with real transit data?
**Answer**: Transit agencies collect Automated Passenger Counter (APC) infrared sensor logs or Automated Fare Collection (AFC) smart-card tap-ins. Simply format the real telemetry into the schema defined in `src/preprocessing.py` (`route_id`, `day_type`, `departure_hour`, `weather`, `nearby_event`, `college_term`, `trip_duration`, `passenger_count`, `crowd_level`), save it as `data/crowdcast_sample_data.csv`, and reload the application. The caching pipeline will automatically re-train on the real data.

---

## 8. License & Attribution

Developed for demonstration and academic evaluation purposes. SYNTHETIC DEMONSTRATION DATA.
"# CrowdCast-AI" 

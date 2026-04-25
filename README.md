# Employee Performance Predictor

An end-to-end machine learning system that predicts employee performance levels — High, Medium, or Low — using a scalable sklearn Pipeline architecture, a Streamlit analytics dashboard, and a FastAPI REST endpoint.

---

## Problem Statement

HR departments in large organizations lack data-driven tools to identify high performers, detect at-risk employees before attrition occurs, and allocate training budgets effectively. This project provides a production-ready ML system to solve all three challenges using structured HR data.

---

## Business Value

| Stakeholder       | Benefit                                              |
|-------------------|------------------------------------------------------|
| HR Manager        | Identify top talent for promotion decisions          |
| Business Leader   | Reduce attrition cost (approx $50K per employee)    |
| L&D Team          | Target training investment where it is needed most  |
| Line Manager      | Get AI-backed coaching recommendations per employee |

---

## Tech Stack

| Category         | Tools                                                        |
|------------------|--------------------------------------------------------------|
| Language         | Python 3.11                                                  |
| Data             | Pandas, NumPy                                                |
| ML Pipeline      | Scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)     |
| Models           | Random Forest, Gradient Boosting, Logistic Regression, SVM, KNN |
| Visualization    | Matplotlib, Seaborn, Plotly                                  |
| Dashboard        | Streamlit                                                    |
| REST API         | FastAPI, Uvicorn, Pydantic                                   |
| Testing          | Pytest (25 unit and integration tests)                       |
| Deployment       | Docker                                                       |

---

## Architecture

```
Raw HR Data (CSV)
      |
      v
FeatureEngineer          adds 4 domain-driven features
      |
      v
ColumnTransformer
  -- StandardScaler      numeric columns
  -- OneHotEncoder       categorical columns
      |
      v
Classifier               best model selected via cross-validation
      |
      v
pipeline.pkl             single serialized artifact
      |
      +-------> Streamlit Dashboard (app.py)
      |
      +-------> FastAPI REST API    (src/api.py)
```

The entire preprocessing and prediction chain lives inside one sklearn Pipeline object. There are no separate scaler files, encoder files, or feature list files to manage. Deploying a new model means replacing one file.

---

## Folder Structure

```
Employee-Performance-Predictor/
|
+-- src/
|   +-- config.py          Single source of truth for all paths, params, constants
|   +-- pipeline.py        Sklearn Pipeline with custom FeatureEngineer transformer
|   +-- generate_data.py   Synthetic HR dataset generator (1000 rows)
|   +-- preprocess.py      Data loading and cleaning only (pipeline handles encoding)
|   +-- train_model.py     Cross-validation, GridSearchCV, model versioning
|   +-- predict.py         Single and batch prediction engine, attrition risk score
|   +-- eda.py             7 EDA charts saved to images/
|   +-- api.py             FastAPI REST endpoint (predict, batch, health, model info)
|
+-- tests/
|   +-- test_pipeline.py   25 pytest unit and integration tests
|
+-- data/
|   +-- hr_dataset.csv     Generated synthetic dataset
|
+-- models/
|   +-- pipeline.pkl       Active trained pipeline (single file replaces 5 old files)
|   +-- versions/          Timestamped backup of every training run
|
+-- outputs/
|   +-- model_comparison.csv
|   +-- model_metadata.json
|   +-- feature_importance.csv
|   +-- classification_report.json
|   +-- confusion_matrix.json
|
+-- images/                7 EDA chart PNG files
+-- app.py                 Streamlit dashboard (9 pages)
+-- main.py                One-command pipeline runner
+-- Dockerfile             Container definition
+-- requirements.txt
+-- README.md
```

### File locations explained

- `src/config.py` — belongs in `src/` because it is imported by every other source module. All modules do `from config import CFG`.
- `src/api.py` — belongs in `src/` because it is part of the source package. Run as `uvicorn src.api:app`.
- `src/pipeline.py` — belongs in `src/` because it defines the core transformer classes used during both training and inference.
- `tests/test_pipeline.py` — belongs in `tests/` following standard Python project convention. Pytest discovers it automatically with `pytest tests/ -v`.

---

## Setup and Installation

### Requirements

- Python 3.10 or higher
- pip

### Install dependencies

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/employee-performance-predictor.git
cd employee-performance-predictor

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac / Linux)
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Train the model

```bash
python main.py
```

This runs all phases in sequence:
1. Generates 1000-row synthetic HR dataset
2. Runs 5-fold stratified cross-validation on 5 ML models
3. Fits the best model and saves `models/pipeline.pkl`
4. Saves a timestamped version backup in `models/versions/`
5. Generates 7 EDA charts in `images/`
6. Prints a demo prediction

Optional flags:

```bash
python main.py --tune             # add GridSearchCV hyperparameter tuning
python main.py --samples 5000     # generate a larger dataset
python main.py --test             # run pytest after training
```

### Step 2 — Launch the dashboard

```bash
streamlit run app.py
```

### Step 3 — Start the REST API (optional)

```bash
# Install API dependencies first
pip install fastapi uvicorn

# Start server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs are available at `http://localhost:8000/docs`

### Step 4 — Run tests

```bash
pytest tests/ -v
```

---

## Dashboard Pages

| Page               | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Analytics          | KPI cards, distribution charts, department breakdown, heatmap        |
| Single Prediction  | Form input, probability bars, attrition risk gauge, HR recommendations |
| What-If Simulator  | Compare before and after intervention side by side with risk delta    |
| Batch Prediction   | Upload CSV, predict all rows, download enriched results              |
| Model Comparison   | CV F1 scores with error bars, training times, feature importances    |
| Confusion Matrix   | Raw counts, normalised percentages, per-class precision/recall/F1    |
| HR Insights        | At-risk employee table, department KPIs, training gap metric         |
| Model Versions     | Browse timestamped pipeline backups, view active metadata            |
| API Guide          | Copy-paste curl and Python examples for all endpoints                |

---

## REST API Endpoints

| Method | Endpoint         | Description                        |
|--------|------------------|------------------------------------|
| GET    | /health          | Health check and pipeline status   |
| GET    | /model/info      | Model metadata and CV results      |
| POST   | /predict         | Single employee prediction         |
| POST   | /predict/batch   | Multiple employees at once         |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32, "gender": "Female", "education": "Master",
    "department": "Engineering", "experience_years": 8,
    "salary": 75000, "training_hours": 60,
    "projects_completed": 12, "avg_monthly_hours": 180,
    "satisfaction_score": 4.2, "last_promotion_years": 2,
    "absenteeism_days": 3, "peer_review_score": 4.0,
    "manager_rating": 4.5
  }'
```

Example response:

```json
{
  "label": "High",
  "confidence": 87.3,
  "probabilities": {"High": 87.3, "Medium": 10.2, "Low": 2.5},
  "risk_score": 1.8,
  "recommendations": [
    "Fast-track promotion candidate",
    "Assign to high-impact strategic projects"
  ]
}
```

---

## Docker Deployment

```bash
# Build image
docker build -t employee-performance-predictor .

# Run Streamlit dashboard
docker run -p 8501:8501 employee-performance-predictor

# Run FastAPI server
docker run -p 8000:8000 employee-performance-predictor \
  uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## ML Models Compared

| Model               | Role                          |
|---------------------|-------------------------------|
| Random Forest       | Primary winner in most runs   |
| Gradient Boosting   | Strong alternative            |
| SVM (RBF kernel)    | Good on balanced classes      |
| Logistic Regression | Fast baseline                 |
| KNN (k=7, weighted) | Distance-weighted baseline    |

Selection is done automatically by 5-fold stratified cross-validation using weighted F1 score as the metric. The best model is then fitted on the full training set and saved.

---

## Engineered Features

Four domain-driven features are added inside the Pipeline's `FeatureEngineer` transformer:

| Feature                 | Formula                                              | Meaning                              |
|-------------------------|------------------------------------------------------|--------------------------------------|
| productivity_ratio      | projects_completed / (avg_monthly_hours / 160)       | Output per normalised work month     |
| engagement_score        | 0.4 x satisfaction + 0.3 x peer_review + 0.3 x manager | Weighted engagement index         |
| career_pace             | experience_years / (age - 21)                        | Speed of career progression          |
| training_effectiveness  | training_hours / 50                                  | Training relative to team average    |

To add a new feature, edit one method in `FeatureEngineer.transform()`. No other file needs to change.

---

## Key Design Decisions

**Single pipeline pickle instead of multiple files**
The old approach required 5 separate files: model, scaler, label encoder, categorical encoder dict, and feature name list. All of these are now encoded in one sklearn Pipeline object. This eliminates desync bugs where the scaler from one run is used with the model from another.

**config.py as single source of truth**
All paths, thresholds, and hyperparameters are defined once in `config.py`. Every module imports `CFG`. Changing the data path or a model parameter requires editing one line.

**ColumnTransformer with remainder=drop**
Any column not listed in `CATEGORICAL_FEATURES` or `NUMERIC_FEATURES` in config is silently dropped. This means raw input can contain extra columns (like employee_id or performance_score) without causing errors.

**class_weight='balanced'**
Tree-based and linear models use balanced class weights to handle the natural imbalance between High, Medium, and Low performers without requiring resampling.

**Model versioning**
Every training run saves a timestamped copy of the pipeline in `models/versions/`. This means any previous model can be loaded and compared without retraining.

---

## Test Suite

```bash
pytest tests/ -v
```

25 tests across 6 categories:

| Category             | Tests |
|----------------------|-------|
| Data generation      | 4     |
| Feature engineering  | 3     |
| Pipeline correctness | 5     |
| Risk score logic     | 3     |
| Live prediction      | 5     |
| Config integrity     | 5     |


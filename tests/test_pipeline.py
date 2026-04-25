"""
tests/test_pipeline.py — Unit + Integration Tests
Run:  pytest tests/ -v
"""
import os, sys, pytest
import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from config import CFG
from generate_data import generate_hr_dataset
from pipeline import build_pipeline, FeatureEngineer, get_all_pipelines
from predict import compute_risk_score

@pytest.fixture(scope="module")
def small_df():
    return generate_hr_dataset(n=200)

@pytest.fixture(scope="module")
def fitted_pipeline(small_df):
    X = small_df.drop(columns=[CFG.TARGET_COL])
    y = small_df[CFG.TARGET_COL].astype(str)
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe

@pytest.fixture
def sample_employee():
    return {"age":32,"gender":"Female","education":"Master","department":"Engineering",
            "experience_years":8,"salary":75000,"training_hours":60,"projects_completed":12,
            "avg_monthly_hours":180,"satisfaction_score":4.2,"last_promotion_years":2,
            "absenteeism_days":3,"peer_review_score":4.0,"manager_rating":4.5,"performance_score":0}

class TestDataGeneration:
    def test_shape(self, small_df):
        assert len(small_df) == 200 and small_df.shape[1] == 17
    def test_no_nulls(self, small_df):
        assert small_df.isnull().sum().sum() == 0
    def test_label_values(self, small_df):
        assert set(small_df["performance_label"].unique()).issubset({"High","Medium","Low"})
    def test_scores_in_range(self, small_df):
        for col in ["satisfaction_score","peer_review_score","manager_rating"]:
            assert small_df[col].between(1.0,5.0).all()

class TestFeatureEngineer:
    def test_new_columns_exist(self, small_df):
        out = FeatureEngineer().transform(small_df)
        for c in ["productivity_ratio","engagement_score","career_pace","training_effectiveness"]:
            assert c in out.columns
    def test_no_nulls(self, small_df):
        out = FeatureEngineer().transform(small_df)
        assert out[["productivity_ratio","engagement_score","career_pace","training_effectiveness"]].isnull().sum().sum() == 0
    def test_idempotent(self, small_df):
        fe = FeatureEngineer()
        r1 = fe.transform(small_df)["engagement_score"].values
        r2 = fe.transform(small_df)["engagement_score"].values
        np.testing.assert_array_equal(r1, r2)

class TestPipeline:
    def test_builds(self):
        pipe = build_pipeline()
        assert all(s in pipe.named_steps for s in ["feature_engineer","preprocessor","classifier"])
    def test_fits_and_predicts(self, fitted_pipeline, small_df):
        X    = small_df.drop(columns=[CFG.TARGET_COL])
        pred = fitted_pipeline.predict(X)
        assert len(pred) == len(X)
        assert all(p in {"High","Low","Medium"} for p in pred)
    def test_proba_sums_to_one(self, fitted_pipeline, small_df):
        X = small_df.drop(columns=[CFG.TARGET_COL])
        np.testing.assert_allclose(fitted_pipeline.predict_proba(X).sum(axis=1), 1.0, atol=1e-6)
    def test_all_five_pipelines(self):
        assert len(get_all_pipelines()) == 5
    def test_minimum_accuracy(self, small_df):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        X = small_df.drop(columns=[CFG.TARGET_COL])
        y = small_df[CFG.TARGET_COL].astype(str)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
        pipe = build_pipeline(); pipe.fit(Xtr,ytr)
        assert accuracy_score(yte,pipe.predict(Xte)) >= 0.60

class TestRiskScore:
    def test_range(self, sample_employee):
        s = compute_risk_score(sample_employee)
        assert 0 <= s <= 10
    def test_high_performer_low_risk(self):
        assert compute_risk_score({"satisfaction_score":5.0,"absenteeism_days":0,"training_hours":90,"manager_rating":5.0}) < 5.0
    def test_low_performer_high_risk(self):
        assert compute_risk_score({"satisfaction_score":1.0,"absenteeism_days":30,"training_hours":0,"manager_rating":1.0}) > 5.0

class TestConfig:
    def test_no_overlap_cat_num(self):
        assert len(set(CFG.CATEGORICAL_FEATURES) & set(CFG.NUMERIC_FEATURES)) == 0
    def test_test_size_valid(self):
        assert 0 < CFG.TEST_SIZE < 1
    def test_cv_folds(self):
        assert CFG.CV_FOLDS >= 2
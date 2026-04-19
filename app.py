"""
app.py — Employee Performance Predictor | Streamlit Dashboard
=============================================================
Run with:  streamlit run app.py

Make sure you have run  python main.py  first to generate:
  data/hr_dataset.csv
  models/best_model.pkl  (+ scaler, encoders)
  outputs/model_comparison.csv
  outputs/feature_importance.csv
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Root-relative path helper ───────────────────────────────────────
# app.py is at project root, so BASE == project root
BASE = os.path.dirname(os.path.abspath(__file__))

def P(*parts):
    """Build an absolute path from project-root-relative parts."""
    return os.path.join(BASE, *parts)

DATA_CSV   = P("data", "hr_dataset.csv")
MODELS_DIR = P("models")
OUT_DIR    = P("outputs")

# ─── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.pred-box {
    padding: 28px 20px; border-radius: 14px; text-align: center;
    background: #1e2130; margin-top: 16px;
    border: 1px solid #2a3050;
}
.rec-row {
    background: #1a2030; border-left: 4px solid #667eea;
    padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
    font-size: 14px;
}
.kpi-card {
    background: #1e2130; border-radius: 12px; padding: 18px;
    border-top: 3px solid #667eea; margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

PERF_COLORS = {"High": "#00d4a7", "Medium": "#ffd166", "Low": "#ef4565"}
DEPT_LIST   = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"]
EDU_LIST    = ["High School", "Bachelor", "Master", "PhD"]
GEN_LIST    = ["Male", "Female", "Other"]
HR_RECS = {
    "High":   ["🌟 Fast-track promotion candidate — present to leadership",
               "🎯 Assign to high-impact strategic projects",
               "🏆 Nominate for quarterly recognition award",
               "📈 Offer leadership development / mentorship track"],
    "Medium": ["📚 Enroll in targeted skill-development training",
               "🤝 Pair with a high-performing mentor for 3 months",
               "🎯 Co-create clear quarterly OKRs with manager",
               "💬 Schedule bi-weekly coaching 1:1 sessions"],
    "Low":    ["🚨 Initiate a formal 90-day Performance Improvement Plan",
               "🔍 Root-cause 1:1 — workload? burnout? role mismatch?",
               "📞 HR well-being check-in & mental health referral",
               "🛠 Immediate structured training + daily manager check-in"],
}


# ─── Cached loaders ──────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_CSV)


@st.cache_resource
def load_model_artifacts():
    def _load(fname):
        fpath = os.path.join(MODELS_DIR, fname)
        with open(fpath, "rb") as f:
            return pickle.load(f)
    model     = _load("best_model.pkl")
    scaler    = _load("scaler.pkl")
    le_target = _load("le_target.pkl")
    le_dict   = _load("le_dict.pkl")
    features  = _load("feature_names.pkl")
    return model, scaler, le_target, le_dict, features


def models_ready():
    return os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl"))


# ─── Feature engineering (must match preprocess.py) ──────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["productivity_ratio"]     = df["projects_completed"] / (df["avg_monthly_hours"] / 160.0)
    df["engagement_score"]       = (0.4 * df["satisfaction_score"]
                                    + 0.3 * df["peer_review_score"]
                                    + 0.3 * df["manager_rating"])
    df["career_pace"]            = df["experience_years"] / (df["age"] - 21).clip(lower=1)
    df["training_effectiveness"] = df["training_hours"] / 50.0   # fixed mean avoids single-row bug
    return df


# ─── Safe categorical encoding (pandas 2.x fix) ──────────────────────
def encode_cats(df: pd.DataFrame, le_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for col, le in le_dict.items():
        if col in df.columns:
            enc = le.transform(df[col].astype(str)).astype(int)
            df[col] = pd.Series(enc, index=df.index, dtype=int)
    return df


# ─── Prediction helper ───────────────────────────────────────────────
def predict_employee(emp: dict):
    model, scaler, le_target, le_dict, features = load_model_artifacts()
    df = pd.DataFrame([emp])
    df = encode_cats(df, le_dict)
    df = engineer_features(df)
    for col in features:                 # add any missing cols as 0
        if col not in df.columns:
            df[col] = 0
    df = df[features]
    X       = scaler.transform(df)
    enc     = model.predict(X)[0]
    label   = le_target.inverse_transform([enc])[0]
    proba   = model.predict_proba(X)[0]
    conf    = round(float(proba.max()) * 100, 1)
    probs   = {c: round(float(p) * 100, 1) for c, p in zip(le_target.classes_, proba)}
    return label, conf, probs


# ─── Sidebar navigation ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 EPP Dashboard")
    st.caption("Employee Performance Predictor")
    st.divider()
    page = st.radio(
        "Go to",
        ["📊 Analytics Dashboard",
         "🔮 Single Prediction",
         "⚙️ What-If Simulator",
         "📁 Batch Prediction",
         "📈 Model Comparison",
         "💡 HR Insights"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Run `python main.py` first to train models.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    st.caption("Live insights from the synthetic HR dataset (1,000 employees).")

    if not os.path.exists(DATA_CSV):
        st.error(f"Dataset not found at `{DATA_CSV}`. Run `python main.py` first.")
        st.stop()

    df = load_dataset()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Employees",        f"{len(df):,}")
    c2.metric("🌟 High",  f"{(df['performance_label']=='High').sum():,}",
              f"{(df['performance_label']=='High').mean()*100:.1f}%")
    c3.metric("⚡ Medium",f"{(df['performance_label']=='Medium').sum():,}",
              f"{(df['performance_label']=='Medium').mean()*100:.1f}%")
    c4.metric("⚠️ Low",   f"{(df['performance_label']=='Low').sum():,}",
              f"{(df['performance_label']=='Low').mean()*100:.1f}%")
    c5.metric("💰 Avg Salary", f"${df['salary'].mean():,.0f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Performance Distribution")
        fig = px.pie(df, names="performance_label", color="performance_label",
                     color_discrete_map=PERF_COLORS, hole=0.45, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Performance by Department")
        dp = df.groupby(["department","performance_label"]).size().reset_index(name="count")
        fig = px.bar(dp, x="department", y="count", color="performance_label",
                     color_discrete_map=PERF_COLORS, barmode="group", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Training Hours by Performance")
        fig = px.box(df, x="performance_label", y="training_hours",
                     color="performance_label", color_discrete_map=PERF_COLORS,
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### Satisfaction vs Performance Score")
        fig = px.scatter(df, x="satisfaction_score", y="performance_score",
                         color="performance_label", color_discrete_map=PERF_COLORS,
                         opacity=0.5, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number).drop(columns=["employee_id"])
    fig = px.imshow(num_df.corr(), text_auto=".2f",
                    color_continuous_scale="RdYlGn", template="plotly_dark", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Single Prediction
# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Single Prediction":
    st.title("🔮 Single Employee Prediction")
    st.caption("Fill in the details below and click Predict.")

    if not models_ready():
        st.error("❌ Model files not found. Please run `python main.py` first, then refresh.")
        st.stop()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 👤 Personal & Role Details")
        age        = st.slider("Age", 22, 60, 32)
        gender     = st.selectbox("Gender", GEN_LIST)
        education  = st.selectbox("Education Level", EDU_LIST, index=1)
        department = st.selectbox("Department", DEPT_LIST)
        experience = st.slider("Experience (years)", 0, 38, 8)
        salary     = st.number_input("Annual Salary (USD)", 25000, 200000, 65000, step=1000)

    with col_b:
        st.markdown("#### 📋 Work Performance Metrics")
        training_hrs = st.slider("Training Hours (this year)", 0, 100, 40)
        projects     = st.slider("Projects Completed", 1, 25, 10)
        monthly_hrs  = st.slider("Avg Monthly Hours", 140, 280, 180)
        satisfaction = st.slider("Satisfaction Score (1–5)", 1.0, 5.0, 3.5, 0.1)
        promo_years  = st.slider("Years Since Last Promotion", 0, 10, 2)
        absent_days  = st.slider("Absenteeism Days", 0, 30, 5)
        peer_score   = st.slider("Peer Review Score (1–5)", 1.0, 5.0, 3.5, 0.1)
        mgr_rating   = st.slider("Manager Rating (1–5)", 1.0, 5.0, 3.8, 0.1)

    st.divider()
    clicked = st.button("🚀 Predict Performance", use_container_width=True, type="primary")

    if clicked:
        emp = {
            "age": age, "gender": gender, "education": education,
            "department": department, "experience_years": experience,
            "salary": salary, "training_hours": training_hrs,
            "projects_completed": projects, "avg_monthly_hours": monthly_hrs,
            "satisfaction_score": satisfaction, "last_promotion_years": promo_years,
            "absenteeism_days": absent_days, "peer_review_score": peer_score,
            "manager_rating": mgr_rating,
        }

        # ── Predict (show full traceback if something goes wrong) ────
        label, confidence, probs = predict_employee(emp)

        color = PERF_COLORS[label]
        emoji = "🌟" if label == "High" else "⚡" if label == "Medium" else "⚠️"

        # ── Result card ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="pred-box">
            <div style="font-size:42px; font-weight:800; color:{color}">
                {emoji} {label} Performer
            </div>
            <div style="color:#aaa; font-size:18px; margin-top:8px">
                Confidence: <strong style="color:white">{confidence}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bar chart ────────────────────────────────────
        st.markdown("#### Probability Breakdown")
        fig = go.Figure()
        for cls in ["High", "Medium", "Low"]:
            prob = probs.get(cls, 0)
            fig.add_trace(go.Bar(
                x=[prob], y=[cls], orientation="h",
                marker_color=PERF_COLORS[cls],
                text=[f"{prob}%"], textposition="inside",
                name=cls,
            ))
        fig.update_layout(
            template="plotly_dark", showlegend=False,
            xaxis=dict(title="Probability (%)", range=[0, 105]),
            height=220, margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── HR Recommendations ───────────────────────────────────────
        st.markdown("#### 💼 HR Recommendations")
        for rec in HR_RECS[label]:
            st.markdown(f'<div class="rec-row">{rec}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — What-If Simulator
# ══════════════════════════════════════════════════════════════════════
elif page == "⚙️ What-If Simulator":
    st.title("⚙️ What-If Simulator")
    st.caption("Compare performance before and after an HR intervention.")

    if not models_ready():
        st.error("❌ Model files not found. Please run `python main.py` first, then refresh.")
        st.stop()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("**Fixed Info**")
        age        = st.number_input("Age", 22, 60, 34)
        gender     = st.selectbox("Gender", GEN_LIST)
        education  = st.selectbox("Education", EDU_LIST, index=1)
        department = st.selectbox("Department", DEPT_LIST)
        experience = st.slider("Experience (yrs)", 0, 38, 10)
        salary     = st.number_input("Salary", 25000, 200000, 60000, step=1000)

    with col2:
        st.markdown("**:red[Before] Intervention**")
        t_b = st.slider("Training Hours",   0, 100, 15, key="tb")
        p_b = st.slider("Projects",         1, 25,   4, key="pb")
        s_b = st.slider("Satisfaction",  1.0, 5.0, 2.4, 0.1, key="sb")
        a_b = st.slider("Absenteeism",      0, 30,  20, key="ab")
        m_b = st.slider("Manager Rating", 1.0, 5.0, 2.5, 0.1, key="mb")

    with col3:
        st.markdown("**:green[After] Intervention**")
        t_a = st.slider("Training Hours",   0, 100, 65, key="ta")
        p_a = st.slider("Projects",         1, 25,  14, key="pa")
        s_a = st.slider("Satisfaction",  1.0, 5.0, 4.1, 0.1, key="sa")
        a_a = st.slider("Absenteeism",      0, 30,   4, key="aa")
        m_a = st.slider("Manager Rating", 1.0, 5.0, 4.4, 0.1, key="ma")

    if st.button("🔄 Compare Before vs After", use_container_width=True, type="primary"):
        base = dict(age=age, gender=gender, education=education, department=department,
                    experience_years=experience, salary=salary,
                    last_promotion_years=2, avg_monthly_hours=180,
                    peer_review_score=3.5)

        emp_before = {**base, "training_hours": t_b, "projects_completed": p_b,
                      "satisfaction_score": s_b, "absenteeism_days": a_b, "manager_rating": m_b}
        emp_after  = {**base, "training_hours": t_a, "projects_completed": p_a,
                      "satisfaction_score": s_a, "absenteeism_days": a_a, "manager_rating": m_a}

        lb, cb, _ = predict_employee(emp_before)
        la, ca, _ = predict_employee(emp_after)

        r1, r2 = st.columns(2)
        with r1:
            cb_color = PERF_COLORS[lb]
            st.markdown(f"""
            <div class="pred-box" style="border-top:4px solid {cb_color}">
                <div style="color:#aaa;font-size:13px">BEFORE</div>
                <div style="font-size:34px;font-weight:800;color:{cb_color}">{lb}</div>
                <div style="color:#aaa">Confidence: {cb}%</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            ca_color = PERF_COLORS[la]
            st.markdown(f"""
            <div class="pred-box" style="border-top:4px solid {ca_color}">
                <div style="color:#aaa;font-size:13px">AFTER</div>
                <div style="font-size:34px;font-weight:800;color:{ca_color}">{la}</div>
                <div style="color:#aaa">Confidence: {ca}%</div>
            </div>""", unsafe_allow_html=True)

        order = {"Low": 0, "Medium": 1, "High": 2}
        delta = order[la] - order[lb]
        if delta > 0:
            st.success(f"⬆️ Intervention worked! Performance improved: **{lb} → {la}**")
        elif delta < 0:
            st.error(f"⬇️ Performance declined: **{lb} → {la}**")
        else:
            st.info(f"➡️ No change in performance level: stays **{la}**")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction")
    st.caption("Upload a CSV file and get predictions for every employee.")

    if not models_ready():
        st.error("❌ Model files not found. Please run `python main.py` first, then refresh.")
        st.stop()

    st.info("**Required columns:** age, gender, education, department, experience_years, "
            "salary, training_hours, projects_completed, avg_monthly_hours, satisfaction_score, "
            "last_promotion_years, absenteeism_days, peer_review_score, manager_rating")

    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"**Loaded:** {len(df_up)} rows")
        st.dataframe(df_up.head(), use_container_width=True)

        if st.button("⚡ Run Batch Prediction", type="primary"):
            model, scaler, le_target, le_dict, features = load_model_artifacts()
            df_proc = encode_cats(df_up.copy(), le_dict)
            df_proc = engineer_features(df_proc)
            for col in features:
                if col not in df_proc.columns:
                    df_proc[col] = 0
            df_proc = df_proc[features]
            X     = scaler.transform(df_proc)
            preds = le_target.inverse_transform(model.predict(X))
            confs = (model.predict_proba(X).max(axis=1) * 100).round(1)

            df_up["Predicted_Performance"] = preds
            df_up["Confidence_%"]          = confs

            st.success(f"✅ Done! {len(df_up)} predictions made.")
            st.dataframe(df_up, use_container_width=True)

            fig = px.pie(df_up, names="Predicted_Performance",
                         color="Predicted_Performance",
                         color_discrete_map=PERF_COLORS, hole=0.4, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            csv_bytes = df_up.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", csv_bytes,
                               "batch_predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    st.caption("Accuracy and F1 comparison of all 5 trained classifiers.")

    comp_path = os.path.join(OUT_DIR, "model_comparison.csv")
    meta_path = os.path.join(OUT_DIR, "model_metadata.json")
    fi_path   = os.path.join(OUT_DIR, "feature_importance.csv")

    if not os.path.exists(comp_path):
        st.error("❌ No model data found. Please run `python main.py` first, then refresh this page.")
        st.stop()

    df_comp = pd.read_csv(comp_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    st.success(
        f"🏆 Best Model: **{meta['best_model']}**  |  "
        f"Accuracy: **{meta['accuracy']*100:.1f}%**  |  "
        f"F1 Score: **{meta['f1_score']*100:.1f}%**"
    )

    st.markdown("#### Accuracy by Model")
    fig = px.bar(
        df_comp.sort_values("Accuracy", ascending=False),
        x="Model", y="Accuracy",
        color="Accuracy", color_continuous_scale="Viridis",
        template="plotly_dark", text="Accuracy",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 1.15], margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### F1 Score by Model")
    fig2 = px.bar(
        df_comp.sort_values("F1_Score", ascending=False),
        x="Model", y="F1_Score",
        color="F1_Score", color_continuous_scale="Plasma",
        template="plotly_dark", text="F1_Score",
    )
    fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig2.update_layout(yaxis_range=[0, 1.15], margin=dict(t=30))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Full Results Table")
    st.dataframe(df_comp, use_container_width=True)

    if os.path.exists(fi_path):
        st.markdown("#### 🔑 Top Feature Importances")
        fi = pd.read_csv(fi_path).head(12).sort_values("Importance")
        fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Turbo",
                      template="plotly_dark")
        fig3.update_layout(margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — HR Insights
# ══════════════════════════════════════════════════════════════════════
elif page == "💡 HR Insights":
    st.title("💡 HR Strategic Insights")
    st.caption("Department-level KPIs and at-risk employee identification.")

    if not os.path.exists(DATA_CSV):
        st.error(f"Dataset not found. Run `python main.py` first.")
        st.stop()

    df = load_dataset()

    dept_sel = st.multiselect("Filter by Department", DEPT_LIST, default=DEPT_LIST)
    df_f = df[df["department"].isin(dept_sel)] if dept_sel else df

    at_risk = df_f[(df_f["satisfaction_score"] < 2.5) & (df_f["absenteeism_days"] > 18)]

    c1, c2, c3 = st.columns(3)
    c1.metric("🚨 At-Risk Employees", len(at_risk),
              f"{len(at_risk)/max(len(df_f),1)*100:.1f}% of workforce")
    c2.metric("💸 Est. Attrition Cost",
              f"${len(at_risk)*50000:,.0f}", "@ $50K per employee")
    c3.metric("📚 Avg Training Gap",
              f"{int(df_f[df_f['performance_label']=='High']['training_hours'].mean() - df_f[df_f['performance_label']=='Low']['training_hours'].mean())} hrs",
              "High vs Low performers")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Avg Training Hours by Department")
        fig = px.bar(
            df_f.groupby("department")["training_hours"].mean().reset_index(),
            x="department", y="training_hours",
            color="training_hours", color_continuous_scale="Blues",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Absenteeism vs Satisfaction")
        fig = px.scatter(df_f, x="absenteeism_days", y="satisfaction_score",
                         color="performance_label", color_discrete_map=PERF_COLORS,
                         opacity=0.6, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🚨 At-Risk Employees (Low Satisfaction AND High Absenteeism)")
    if len(at_risk) > 0:
        st.dataframe(
            at_risk[["employee_id", "department", "satisfaction_score",
                     "absenteeism_days", "performance_label"]]
            .sort_values("satisfaction_score")
            .reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("No at-risk employees in the selected departments.")

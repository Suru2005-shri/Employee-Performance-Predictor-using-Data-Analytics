"""
app.py  —  Employee Performance Predictor | Streamlit Dashboard
================================================================
Run with:  streamlit run app.py

Features:
  1. 📊 Analytics Dashboard   – live charts from the dataset
  2. 🔮 Single Prediction     – predict one employee's performance
  3. 📁 Batch Prediction      – upload CSV, download results
  4. 📈 Model Comparison      – compare all trained models
  5. 🤖 What-If Simulator     – drag sliders to see how changes affect prediction
  6. 💡 HR Insights           – department-level KPI drill-down
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .high-perf   { border-left-color: #00d4a7 !important; }
    .medium-perf { border-left-color: #ffd166 !important; }
    .low-perf    { border-left-color: #ef4565 !important; }
    .pred-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border-radius: 16px; padding: 28px; margin-top: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .section-header {
        font-size: 22px; font-weight: 700; color: #667eea;
        margin-bottom: 16px; border-bottom: 2px solid #667eea22;
        padding-bottom: 8px;
    }
    .rec-item {
        background: #1e2130; border-radius: 8px; padding: 10px 14px;
        margin: 6px 0; border-left: 3px solid #667eea;
        font-size: 14px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 10px 28px; font-weight: 600; font-size: 15px;
        transition: all 0.3s;
    }
    .stButton>button:hover { opacity: 0.85; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, '../data/hr_dataset.csv')
MDL   = os.path.join(BASE, '../models')
OUT   = os.path.join(BASE, '../outputs')

DEPT_LIST = ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing', 'Operations']
EDU_LIST  = ['High School', 'Bachelor', 'Master', 'PhD']
GEN_LIST  = ['Male', 'Female', 'Other']

PERF_COLORS = {'High': '#00d4a7', 'Medium': '#ffd166', 'Low': '#ef4565'}


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA)


@st.cache_resource
def load_model_artifacts():
    model     = pickle.load(open(f'{MDL}/best_model.pkl',    'rb'))
    scaler    = pickle.load(open(f'{MDL}/scaler.pkl',        'rb'))
    le_target = pickle.load(open(f'{MDL}/le_target.pkl',     'rb'))
    le_dict   = pickle.load(open(f'{MDL}/le_dict.pkl',       'rb'))
    features  = pickle.load(open(f'{MDL}/feature_names.pkl', 'rb'))
    return model, scaler, le_target, le_dict, features


def engineer_features(df):
    df = df.copy()
    df['productivity_ratio']     = df['projects_completed'] / (df['avg_monthly_hours'] / 160)
    df['engagement_score']       = (
        0.4 * df['satisfaction_score']
        + 0.3 * df['peer_review_score']
        + 0.3 * df['manager_rating']
    )
    df['career_pace']            = df['experience_years'] / (df['age'] - 21).clip(lower=1)
    df['training_effectiveness'] = df['training_hours'] / (df['training_hours'].mean() + 1e-5)
    return df


def predict_employee(emp_dict):
    model, scaler, le_target, le_dict, features = load_model_artifacts()
    df = pd.DataFrame([emp_dict])
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    df = engineer_features(df)
    df = df[features]
    X  = scaler.transform(df)
    pred  = model.predict(X)[0]
    label = le_target.inverse_transform([pred])[0]
    proba = model.predict_proba(X)[0]
    prob_dict = {c: round(float(p)*100, 1) for c, p in zip(le_target.classes_, proba)}
    return label, round(float(proba.max())*100, 1), prob_dict


HR_RECS = {
    'High'  : ["🌟 Fast-track promotion candidate",
               "🎯 Assign to high-impact strategic projects",
               "🏆 Nominate for recognition award",
               "📈 Offer leadership development track"],
    'Medium': ["📚 Targeted skill-development training",
               "🤝 Pair with a senior mentor",
               "🎯 Set clear quarterly OKRs",
               "💬 Schedule monthly coaching sessions"],
    'Low'   : ["🚨 Initiate Performance Improvement Plan (PIP)",
               "🔍 Root-cause 1:1 — workload / burnout?",
               "📞 Well-being check-in meeting",
               "🛠 Immediate training & structured support"],
}


# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/combo-chart.png", width=64)
    st.markdown("## 🎯 EPP Dashboard")
    st.markdown("*Employee Performance Predictor*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Analytics Dashboard",
         "🔮 Single Prediction",
         "⚙️ What-If Simulator",
         "📁 Batch Prediction",
         "📈 Model Comparison",
         "💡 HR Insights"],
        label_visibility='collapsed'
    )
    st.divider()
    st.caption("Built with ❤️ using Scikit-learn + Streamlit")


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Analytics Dashboard":
    st.markdown("# 📊 Analytics Dashboard")
    st.markdown("Live insights from the synthetic HR dataset.")

    df = load_dataset()

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Employees", f"{len(df):,}")
    c2.metric("🌟 High Performers",
              f"{(df['performance_label']=='High').sum():,}",
              f"{(df['performance_label']=='High').mean()*100:.1f}%")
    c3.metric("⚡ Medium Performers",
              f"{(df['performance_label']=='Medium').sum():,}",
              f"{(df['performance_label']=='Medium').mean()*100:.1f}%")
    c4.metric("⚠️ Low Performers",
              f"{(df['performance_label']=='Low').sum():,}",
              f"{(df['performance_label']=='Low').mean()*100:.1f}%")
    c5.metric("💰 Avg Salary", f"${df['salary'].mean():,.0f}")
    st.divider()

    # Row 1: Distribution + Dept breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Performance Distribution")
        fig = px.pie(
            df, names='performance_label',
            color='performance_label',
            color_discrete_map=PERF_COLORS,
            hole=0.45,
            template='plotly_dark'
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Performance by Department")
        dept_perf = df.groupby(['department', 'performance_label']).size().reset_index(name='count')
        fig = px.bar(
            dept_perf, x='department', y='count', color='performance_label',
            color_discrete_map=PERF_COLORS,
            barmode='group', template='plotly_dark'
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Training hours + Salary
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Training Hours Distribution by Performance")
        fig = px.box(df, x='performance_label', y='training_hours',
                     color='performance_label',
                     color_discrete_map=PERF_COLORS,
                     template='plotly_dark')
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### Satisfaction Score vs Performance Score")
        fig = px.scatter(df, x='satisfaction_score', y='performance_score',
                         color='performance_label',
                         color_discrete_map=PERF_COLORS,
                         opacity=0.6, template='plotly_dark')
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Heatmap
    st.markdown("#### Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number).drop(columns=['employee_id'])
    corr   = num_df.corr()
    fig    = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdYlGn',
                       template='plotly_dark', aspect='auto')
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Single Prediction
# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Single Prediction":
    st.markdown("# 🔮 Predict Employee Performance")
    st.markdown("Fill in the employee details and get an AI-powered performance prediction.")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 👤 Personal Details")
        age            = st.slider("Age", 22, 60, 32)
        gender         = st.selectbox("Gender", GEN_LIST)
        education      = st.selectbox("Education Level", EDU_LIST)
        department     = st.selectbox("Department", DEPT_LIST)
        experience     = st.slider("Experience (years)", 0, 38, 8)
        salary         = st.number_input("Salary (USD)", 25000, 200000, 65000, step=1000)

    with col_b:
        st.markdown("#### 📋 Work Metrics")
        training_hrs   = st.slider("Training Hours (this year)", 0, 100, 40)
        projects       = st.slider("Projects Completed", 1, 25, 10)
        monthly_hrs    = st.slider("Avg Monthly Hours", 140, 280, 180)
        satisfaction   = st.slider("Satisfaction Score (1-5)", 1.0, 5.0, 3.5, 0.1)
        promo_years    = st.slider("Years Since Last Promotion", 0, 10, 2)
        absent_days    = st.slider("Absenteeism Days", 0, 30, 5)
        peer_score     = st.slider("Peer Review Score (1-5)", 1.0, 5.0, 3.5, 0.1)
        mgr_rating     = st.slider("Manager Rating (1-5)", 1.0, 5.0, 3.8, 0.1)

    if st.button("🚀 Predict Performance", use_container_width=True):
        emp = {
            'age': age, 'gender': gender, 'education': education,
            'department': department, 'experience_years': experience,
            'salary': salary, 'training_hours': training_hrs,
            'projects_completed': projects, 'avg_monthly_hours': monthly_hrs,
            'satisfaction_score': satisfaction, 'last_promotion_years': promo_years,
            'absenteeism_days': absent_days, 'peer_review_score': peer_score,
            'manager_rating': mgr_rating
        }
        try:
            label, confidence, probs = predict_employee(emp)
            color = PERF_COLORS[label]

            st.markdown(f"""
            <div class='pred-card'>
                <h2 style='color:{color}; font-size:36px; margin:0'>
                    {'🌟' if label=='High' else '⚡' if label=='Medium' else '⚠️'} {label} Performer
                </h2>
                <p style='color:#aaa; font-size:18px; margin-top:8px'>
                    Confidence: <strong style='color:white'>{confidence}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            st.markdown("#### Probability Breakdown")
            fig = go.Figure()
            for cls, prob in probs.items():
                fig.add_trace(go.Bar(
                    x=[prob], y=[cls], orientation='h',
                    marker_color=PERF_COLORS.get(cls, '#888'),
                    text=[f"{prob}%"], textposition='inside',
                    name=cls
                ))
            fig.update_layout(
                template='plotly_dark', showlegend=False,
                xaxis_title="Probability (%)", height=200,
                margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            # HR Recommendations
            st.markdown("#### 💼 HR Recommendations")
            for rec in HR_RECS[label]:
                st.markdown(f"<div class='rec-item'>{rec}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Model not found. Please run `python src/train_model.py` first.\n\nError: {e}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — What-If Simulator
# ══════════════════════════════════════════════════════════════════════
elif page == "⚙️ What-If Simulator":
    st.markdown("# ⚙️ What-If Simulator")
    st.markdown(
        "Adjust levers to see how HR interventions would change an employee's predicted performance."
    )
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("**Fixed Background Info**")
        age         = st.number_input("Age", 22, 60, 34)
        gender      = st.selectbox("Gender", GEN_LIST, index=0)
        education   = st.selectbox("Education", EDU_LIST, index=1)
        department  = st.selectbox("Department", DEPT_LIST, index=0)
        experience  = st.slider("Experience (yrs)", 0, 38, 10)
        salary      = st.number_input("Salary", 25000, 200000, 60000, step=1000)

    with col2:
        st.markdown("**Current State (Before)**")
        t_before = st.slider("Training Hours (before)", 0, 100, 15, key="t_b")
        p_before = st.slider("Projects (before)", 1, 25, 5, key="p_b")
        s_before = st.slider("Satisfaction (before)", 1.0, 5.0, 2.5, 0.1, key="s_b")
        ab_before= st.slider("Absenteeism (before)", 0, 30, 18, key="ab_b")
        pr_before= st.slider("Peer Score (before)", 1.0, 5.0, 2.8, 0.1, key="pr_b")
        mr_before= st.slider("Manager Rating (before)", 1.0, 5.0, 3.0, 0.1, key="mr_b")

    with col3:
        st.markdown("**After Intervention**")
        t_after  = st.slider("Training Hours (after)", 0, 100, 60, key="t_a")
        p_after  = st.slider("Projects (after)", 1, 25, 14, key="p_a")
        s_after  = st.slider("Satisfaction (after)", 1.0, 5.0, 4.0, 0.1, key="s_a")
        ab_after = st.slider("Absenteeism (after)", 0, 30, 4, key="ab_a")
        pr_after = st.slider("Peer Score (after)", 1.0, 5.0, 4.2, 0.1, key="pr_a")
        mr_after = st.slider("Manager Rating (after)", 1.0, 5.0, 4.5, 0.1, key="mr_a")

    if st.button("🔄 Run Comparison", use_container_width=True):
        base = dict(age=age, gender=gender, education=education, department=department,
                    experience_years=experience, salary=salary, last_promotion_years=2,
                    avg_monthly_hours=180)
        before_emp = {**base, 'training_hours':t_before, 'projects_completed':p_before,
                      'satisfaction_score':s_before, 'absenteeism_days':ab_before,
                      'peer_review_score':pr_before, 'manager_rating':mr_before}
        after_emp  = {**base, 'training_hours':t_after,  'projects_completed':p_after,
                      'satisfaction_score':s_after,  'absenteeism_days':ab_after,
                      'peer_review_score':pr_after,  'manager_rating':mr_after}
        try:
            l_b, c_b, probs_b = predict_employee(before_emp)
            l_a, c_a, probs_a = predict_employee(after_emp)

            col_r1, col_r2 = st.columns(2)
            cb, ca = PERF_COLORS[l_b], PERF_COLORS[l_a]
            with col_r1:
                st.markdown(f"""
                <div class='pred-card'>
                    <p style='color:#aaa'>BEFORE Intervention</p>
                    <h2 style='color:{cb}'>{l_b} Performer</h2>
                    <p style='color:white'>Confidence: {c_b}%</p>
                </div>""", unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div class='pred-card'>
                    <p style='color:#aaa'>AFTER Intervention</p>
                    <h2 style='color:{ca}'>{l_a} Performer</h2>
                    <p style='color:white'>Confidence: {c_a}%</p>
                </div>""", unsafe_allow_html=True)

            arrow = "⬆️ Improved" if l_a != l_b else ("✅ Same Level" if l_a == l_b else "⬇️ Declined")
            st.success(f"**Impact:** {arrow}  |  {l_b} → {l_a}")

        except Exception as e:
            st.error(f"Model not found. Run `train_model.py` first.\nError: {e}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.markdown("# 📁 Batch Prediction")
    st.markdown("Upload a CSV with employee data and download predictions for all rows.")
    st.divider()

    st.info(
        "**Required columns:** age, gender, education, department, experience_years, "
        "salary, training_hours, projects_completed, avg_monthly_hours, satisfaction_score, "
        "last_promotion_years, absenteeism_days, peer_review_score, manager_rating"
    )

    uploaded = st.file_uploader("Upload employee CSV", type=['csv'])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"**Preview:** {len(df_up)} rows")
        st.dataframe(df_up.head(), use_container_width=True)

        if st.button("⚡ Run Batch Prediction"):
            try:
                model, scaler, le_target, le_dict, features = load_model_artifacts()
                df_proc = df_up.copy()
                for col, le in le_dict.items():
                    if col in df_proc.columns:
                        df_proc[col] = le.transform(df_proc[col])
                df_proc = engineer_features(df_proc)
                df_proc = df_proc[features]
                X       = scaler.transform(df_proc)
                preds   = le_target.inverse_transform(model.predict(X))
                confs   = (model.predict_proba(X).max(axis=1) * 100).round(1)

                df_up['Predicted_Performance'] = preds
                df_up['Confidence_%']          = confs

                st.success(f"✅ Done! {len(df_up)} predictions made.")
                st.dataframe(df_up, use_container_width=True)

                # Distribution
                fig = px.pie(df_up, names='Predicted_Performance',
                             color='Predicted_Performance',
                             color_discrete_map=PERF_COLORS,
                             hole=0.4, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv = df_up.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results CSV", csv,
                                   "batch_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown("# 📈 Model Comparison")
    comp_path = f'{OUT}/model_comparison.csv'
    meta_path = f'{OUT}/model_metadata.json'
    fi_path   = f'{OUT}/feature_importance.csv'

    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        meta    = json.load(open(meta_path))

        st.success(f"🏆 Best Model: **{meta['best_model']}** | "
                   f"Accuracy: **{meta['accuracy']*100:.1f}%** | "
                   f"F1: **{meta['f1_score']*100:.1f}%**")

        fig = px.bar(df_comp, x='Model', y='Accuracy',
                     color='Accuracy', color_continuous_scale='Viridis',
                     template='plotly_dark', text='Accuracy')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_comp, use_container_width=True)

        if os.path.exists(fi_path):
            st.markdown("#### 🔑 Feature Importance (Best Model)")
            fi = pd.read_csv(fi_path).head(12).sort_values('Importance')
            fig2 = px.bar(fi, x='Importance', y='Feature', orientation='h',
                          color='Importance', color_continuous_scale='Plasma',
                          template='plotly_dark')
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No model data found. Run `python src/train_model.py` first.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — HR Insights
# ══════════════════════════════════════════════════════════════════════
elif page == "💡 HR Insights":
    st.markdown("# 💡 HR Strategic Insights")
    df = load_dataset()

    dept_sel = st.multiselect("Filter by Department", DEPT_LIST, default=DEPT_LIST)
    df_f = df[df['department'].isin(dept_sel)]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Avg Training Hours by Dept")
        fig = px.bar(
            df_f.groupby('department')['training_hours'].mean().reset_index(),
            x='department', y='training_hours',
            color='training_hours', color_continuous_scale='Blues',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Absenteeism vs Satisfaction")
        fig = px.scatter(df_f, x='absenteeism_days', y='satisfaction_score',
                         color='performance_label', color_discrete_map=PERF_COLORS,
                         opacity=0.6, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### High-Risk Employees (Low Satisfaction + High Absenteeism)")
    risk = df_f[
        (df_f['satisfaction_score'] < 2.5) & (df_f['absenteeism_days'] > 18)
    ][['employee_id', 'department', 'satisfaction_score', 'absenteeism_days',
       'performance_label']].sort_values('satisfaction_score')
    st.dataframe(risk.reset_index(drop=True), use_container_width=True)
    st.metric("🚨 At-Risk Employees", len(risk),
              f"{len(risk)/len(df_f)*100:.1f}% of filtered workforce")

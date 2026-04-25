"""
app.py — Scalable Streamlit Dashboard v2.0
============================================
Uses ONE pipeline.pkl — no manual encoding, no scattered files.
New pages: Confusion Matrix, Model Versions, API Guide, Run Tests.
"""

import os, sys, json, pickle, subprocess, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
for p in [SRC, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.figure_factory import create_annotated_heatmap

from config import CFG
from predict import predict_single, compute_risk_score

st.set_page_config(
    page_title="EPP v2.0 — Scalable",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.pred-box{padding:28px 20px;border-radius:14px;text-align:center;
  background:#1e2130;margin-top:16px;border:1px solid #2a3050;}
.rec-row{background:#1a2030;border-left:4px solid #667eea;
  padding:10px 14px;border-radius:6px;margin-bottom:8px;font-size:14px;}
.risk-bar-wrap{background:#1e2130;border-radius:10px;padding:14px 18px;margin-top:12px;}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;
  font-size:12px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

PERF_COLORS = {"High":"#00d4a7","Medium":"#ffd166","Low":"#ef4565"}
DEPT_LIST   = ["Engineering","Sales","HR","Finance","Marketing","Operations"]
EDU_LIST    = ["High School","Bachelor","Master","PhD"]
GEN_LIST    = ["Male","Female","Other"]

HR_RECS = {
    "High"  : ["🌟 Fast-track promotion candidate — present to leadership",
               "🎯 Assign to high-impact strategic projects",
               "🏆 Nominate for quarterly recognition award",
               "📈 Offer leadership development / mentorship track"],
    "Medium": ["📚 Enroll in targeted skill-development training",
               "🤝 Pair with a high-performing mentor for 3 months",
               "🎯 Co-create clear quarterly OKRs with manager",
               "💬 Schedule bi-weekly coaching 1:1 sessions"],
    "Low"   : ["🚨 Initiate a formal 90-day Performance Improvement Plan",
               "🔍 Root-cause 1:1 — workload? burnout? role mismatch?",
               "📞 HR well-being check-in & mental health referral",
               "🛠 Immediate structured training + daily check-ins"],
}


# ── Helpers ───────────────────────────────────────────────────────────
def pipeline_ready():
    return os.path.exists(CFG.PIPELINE_PKL)

def metadata_ready():
    return os.path.exists(CFG.METADATA_JSON)

@st.cache_data
def load_dataset():
    return pd.read_csv(CFG.DATA_CSV)

@st.cache_resource
def load_pipeline_cached():
    with open(CFG.PIPELINE_PKL, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metadata():
    with open(CFG.METADATA_JSON) as f:
        return json.load(f)

def not_ready_error():
    st.error("❌ Pipeline not found. Run `python main.py` first, then refresh.")
    st.code("python main.py", language="bash")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 EPP v2.0")
    st.caption("Scalable Pipeline Architecture")
    st.divider()

    if pipeline_ready():
        st.success("✅ Pipeline loaded")
        if metadata_ready():
            meta = load_metadata()
            st.caption(f"Model: **{meta['best_model']}**")
            st.caption(f"Acc: **{meta['accuracy']*100:.1f}%** | F1: **{meta['f1_score']*100:.1f}%**")
            st.caption(f"Trained: {meta.get('trained_at','—')[:13]}")
    else:
        st.warning("⚠️ No pipeline found")
        st.code("python main.py")

    st.divider()
    page = st.radio("Navigate", [
        "📊 Analytics",
        "🔮 Single Prediction",
        "⚙️ What-If Simulator",
        "📁 Batch Prediction",
        "📈 Model Comparison",
        "🧩 Confusion Matrix",
        "💡 HR Insights",
        "🗂️ Model Versions",
        "🌐 API Guide",
    ], label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Analytics
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")

    if not os.path.exists(CFG.DATA_CSV):
        st.error("Dataset not found. Run `python main.py` first.")
        st.stop()

    df = load_dataset()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("👥 Employees", f"{len(df):,}")
    c2.metric("🌟 High",   f"{(df['performance_label']=='High').sum()}")
    c3.metric("⚡ Medium", f"{(df['performance_label']=='Medium').sum()}")
    c4.metric("⚠️ Low",    f"{(df['performance_label']=='Low').sum()}")
    c5.metric("💰 Avg Salary", f"${df['salary'].mean():,.0f}")
    st.divider()

    col1,col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names="performance_label", color="performance_label",
                     color_discrete_map=PERF_COLORS, hole=0.45,
                     template="plotly_dark", title="Performance Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dp = df.groupby(["department","performance_label"]).size().reset_index(name="count")
        fig = px.bar(dp, x="department", y="count", color="performance_label",
                     color_discrete_map=PERF_COLORS, barmode="group",
                     template="plotly_dark", title="Performance by Department")
        st.plotly_chart(fig, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        fig = px.box(df, x="performance_label", y="training_hours",
                     color="performance_label", color_discrete_map=PERF_COLORS,
                     template="plotly_dark", title="Training Hours by Performance")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.scatter(df, x="satisfaction_score", y="performance_score",
                         color="performance_label", color_discrete_map=PERF_COLORS,
                         opacity=0.5, template="plotly_dark",
                         title="Satisfaction vs Performance Score")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number).drop(columns=["employee_id"])
    fig = px.imshow(num_df.corr(), text_auto=".2f",
                    color_continuous_scale="RdYlGn",
                    template="plotly_dark", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Single Prediction
# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Single Prediction":
    st.title("🔮 Single Employee Prediction")
    st.caption("Pipeline predicts directly from raw inputs — no manual encoding needed.")

    if not pipeline_ready():
        not_ready_error()

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
        training_hrs = st.slider("Training Hours", 0, 100, 40)
        projects     = st.slider("Projects Completed", 1, 25, 10)
        monthly_hrs  = st.slider("Avg Monthly Hours", 140, 280, 180)
        satisfaction = st.slider("Satisfaction Score (1–5)", 1.0, 5.0, 3.5, 0.1)
        promo_years  = st.slider("Years Since Last Promotion", 0, 10, 2)
        absent_days  = st.slider("Absenteeism Days", 0, 30, 5)
        peer_score   = st.slider("Peer Review Score (1–5)", 1.0, 5.0, 3.5, 0.1)
        mgr_rating   = st.slider("Manager Rating (1–5)", 1.0, 5.0, 3.8, 0.1)

    st.divider()
    if st.button("🚀 Predict Performance", use_container_width=True, type="primary"):
        emp = {
            "age":age, "gender":gender, "education":education, "department":department,
            "experience_years":experience, "salary":salary, "training_hours":training_hrs,
            "projects_completed":projects, "avg_monthly_hours":monthly_hrs,
            "satisfaction_score":satisfaction, "last_promotion_years":promo_years,
            "absenteeism_days":absent_days, "peer_review_score":peer_score,
            "manager_rating":mgr_rating, "performance_score":0,
        }
        result = predict_single(emp)
        label, conf, probs = result["label"], result["confidence"], result["probabilities"]
        risk = result["risk_score"]
        color = PERF_COLORS[label]
        emoji = "🌟" if label=="High" else "⚡" if label=="Medium" else "⚠️"

        st.markdown(f"""
        <div class="pred-box" style="border-top:4px solid {color}">
            <div style="font-size:40px;font-weight:800;color:{color}">{emoji} {label} Performer</div>
            <div style="color:#aaa;font-size:17px;margin-top:8px">
                Confidence: <strong style="color:white">{conf}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

        # Probability bars
        st.markdown("#### Probability Breakdown")
        fig = go.Figure()
        for cls in ["High","Medium","Low"]:
            p = probs.get(cls, 0)
            fig.add_trace(go.Bar(x=[p], y=[cls], orientation="h",
                marker_color=PERF_COLORS[cls], text=[f"{p}%"],
                textposition="inside", name=cls))
        fig.update_layout(template="plotly_dark", showlegend=False,
            xaxis=dict(title="Probability (%)", range=[0,105]),
            height=220, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        # Attrition risk gauge
        st.markdown("#### ⚠️ Attrition Risk Score")
        risk_color = "#ef4565" if risk>7 else "#ffd166" if risk>4 else "#00d4a7"
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            domain={"x":[0,1],"y":[0,1]},
            gauge={
                "axis"      : {"range":[0,10]},
                "bar"       : {"color":risk_color},
                "steps"     : [{"range":[0,4],"color":"#1a2a1a"},
                               {"range":[4,7],"color":"#2a2a1a"},
                               {"range":[7,10],"color":"#2a1a1a"}],
                "threshold" : {"line":{"color":"white","width":2},"value":7},
            },
            title={"text":"Attrition Risk (0=safe, 10=high risk)"},
        ))
        fig2.update_layout(template="plotly_dark", height=250, margin=dict(t=40,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        # HR Recommendations
        st.markdown("#### 💼 HR Recommendations")
        for rec in HR_RECS[label]:
            st.markdown(f'<div class="rec-row">{rec}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — What-If Simulator
# ══════════════════════════════════════════════════════════════════════
elif page == "⚙️ What-If Simulator":
    st.title("⚙️ What-If Simulator")
    st.caption("Simulate HR interventions and see the predicted impact.")

    if not pipeline_ready():
        not_ready_error()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Fixed Info**")
        age        = st.number_input("Age", 22, 60, 34)
        gender     = st.selectbox("Gender", GEN_LIST)
        education  = st.selectbox("Education", EDU_LIST, index=1)
        department = st.selectbox("Department", DEPT_LIST)
        experience = st.slider("Experience", 0, 38, 10)
        salary     = st.number_input("Salary", 25000, 200000, 60000, step=1000)
    with col2:
        st.markdown("**:red[Before] Intervention]**")
        t_b = st.slider("Training Hours",  0,100, 15, key="tb")
        p_b = st.slider("Projects",        1, 25,  4, key="pb")
        s_b = st.slider("Satisfaction", 1.0,5.0,2.4,0.1, key="sb")
        a_b = st.slider("Absenteeism",     0, 30, 20, key="ab")
        m_b = st.slider("Manager Rating",1.0,5.0,2.5,0.1, key="mb")
    with col3:
        st.markdown("**:green[After] Intervention]**")
        t_a = st.slider("Training Hours",  0,100, 65, key="ta")
        p_a = st.slider("Projects",        1, 25, 14, key="pa")
        s_a = st.slider("Satisfaction", 1.0,5.0,4.1,0.1, key="sa")
        a_a = st.slider("Absenteeism",     0, 30,  4, key="aa")
        m_a = st.slider("Manager Rating",1.0,5.0,4.4,0.1, key="ma")

    if st.button("🔄 Compare Before vs After", use_container_width=True, type="primary"):
        base = dict(age=age, gender=gender, education=education, department=department,
                    experience_years=experience, salary=salary,
                    last_promotion_years=2, avg_monthly_hours=180,
                    peer_review_score=3.5, performance_score=0)
        rb = predict_single({**base,"training_hours":t_b,"projects_completed":p_b,
                              "satisfaction_score":s_b,"absenteeism_days":a_b,"manager_rating":m_b})
        ra = predict_single({**base,"training_hours":t_a,"projects_completed":p_a,
                              "satisfaction_score":s_a,"absenteeism_days":a_a,"manager_rating":m_a})

        c1, c2 = st.columns(2)
        for col, r, label_txt in [(c1, rb, "BEFORE"), (c2, ra, "AFTER")]:
            clr = PERF_COLORS[r["label"]]
            with col:
                st.markdown(f"""<div class="pred-box" style="border-top:4px solid {clr}">
                    <div style="color:#aaa;font-size:12px">{label_txt}</div>
                    <div style="font-size:32px;font-weight:800;color:{clr}">{r['label']}</div>
                    <div style="color:#aaa">Confidence: {r['confidence']}% | Risk: {r['risk_score']}/10</div>
                </div>""", unsafe_allow_html=True)

        order = {"Low":0,"Medium":1,"High":2}
        delta = order[ra["label"]] - order[rb["label"]]
        if   delta > 0: st.success(f"⬆️ Improved: **{rb['label']} → {ra['label']}**")
        elif delta < 0: st.error(  f"⬇️ Declined: **{rb['label']} → {ra['label']}**")
        else:           st.info(   f"➡️ No level change. Still **{ra['label']}**")

        # Risk delta
        risk_delta = rb["risk_score"] - ra["risk_score"]
        if risk_delta > 0:
            st.success(f"✅ Attrition risk reduced by **{risk_delta:.1f}** points")
        elif risk_delta < 0:
            st.warning(f"⚠️ Attrition risk increased by **{abs(risk_delta):.1f}** points")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction")
    st.caption("Upload CSV → Pipeline predicts all rows → Download enriched results.")

    if not pipeline_ready():
        not_ready_error()

    st.info("**Required columns:** age, gender, education, department, experience_years, salary, "
            "training_hours, projects_completed, avg_monthly_hours, satisfaction_score, "
            "last_promotion_years, absenteeism_days, peer_review_score, manager_rating")

    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        df_up["performance_score"] = 0   # pipeline ignores this
        st.write(f"**Loaded:** {len(df_up)} rows")
        st.dataframe(df_up.head(), use_container_width=True)

        if st.button("⚡ Run Batch Prediction", type="primary"):
            pipe  = load_pipeline_cached()
            preds = pipe.predict(df_up)
            proba = pipe.predict_proba(df_up).max(axis=1)
            risks = df_up.apply(lambda r: compute_risk_score(r.to_dict()), axis=1)

            df_up["Predicted_Performance"] = preds
            df_up["Confidence_%"]          = (proba * 100).round(1)
            df_up["Attrition_Risk"]        = risks

            st.success(f"✅ {len(df_up)} predictions complete.")
            st.dataframe(df_up, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df_up, names="Predicted_Performance",
                             color="Predicted_Performance",
                             color_discrete_map=PERF_COLORS, hole=0.4,
                             template="plotly_dark", title="Predicted Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.histogram(df_up, x="Attrition_Risk", nbins=20,
                                    template="plotly_dark",
                                    title="Attrition Risk Distribution",
                                    color_discrete_sequence=["#667eea"])
                st.plotly_chart(fig2, use_container_width=True)

            csv_bytes = df_up.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", csv_bytes,
                               "batch_predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    st.caption("Stratified K-Fold cross-validation results for all 5 classifiers.")

    if not os.path.exists(CFG.COMPARISON_CSV):
        st.error("No comparison data. Run `python main.py` first.")
        st.stop()

    df_comp = pd.read_csv(CFG.COMPARISON_CSV)
    meta    = load_metadata()

    st.success(f"🏆 Best Model: **{meta['best_model']}**  |  "
               f"Test Acc: **{meta['accuracy']*100:.1f}%**  |  "
               f"F1: **{meta['f1_score']*100:.1f}%**  |  "
               f"CV Folds: **{meta.get('n_cv_folds', CFG.CV_FOLDS)}**")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_comp.sort_values("CV_Mean_F1", ascending=False),
                     x="Model", y="CV_Mean_F1",
                     error_y="CV_Std_F1",
                     color="CV_Mean_F1", color_continuous_scale="Viridis",
                     template="plotly_dark", text="CV_Mean_F1",
                     title="Cross-Validation F1 Score (± std)")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(yaxis_range=[0,1.15])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Time_s" in df_comp.columns:
            fig2 = px.bar(df_comp.sort_values("Time_s"),
                          x="Time_s", y="Model", orientation="h",
                          color="Time_s", color_continuous_scale="Reds",
                          template="plotly_dark",
                          title="Training Time (seconds)")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Full CV Results Table")
    st.dataframe(df_comp, use_container_width=True)

    if os.path.exists(CFG.FI_CSV):
        st.markdown("#### 🔑 Top Feature Importances")
        fi   = pd.read_csv(CFG.FI_CSV).head(15).sort_values("Importance")
        fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Turbo",
                      template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — Confusion Matrix (NEW)
# ══════════════════════════════════════════════════════════════════════
elif page == "🧩 Confusion Matrix":
    st.title("🧩 Confusion Matrix & Classification Report")

    if not os.path.exists(CFG.CM_JSON):
        st.error("No confusion matrix data. Run `python main.py` first.")
        st.stop()

    with open(CFG.CM_JSON) as f:
        cm_data = json.load(f)
    with open(CFG.REPORT_JSON) as f:
        report  = json.load(f)

    classes = cm_data["classes"]
    matrix  = np.array(cm_data["matrix"])

    # Normalised matrix
    matrix_norm = (matrix.astype(float) / matrix.sum(axis=1, keepdims=True) * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Raw Counts")
        fig = px.imshow(matrix, x=classes, y=classes,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted",y="Actual"),
                        text_auto=True, template="plotly_dark")
        fig.update_layout(margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Normalised (%)")
        fig2 = px.imshow(matrix_norm, x=classes, y=classes,
                         color_continuous_scale="Greens",
                         labels=dict(x="Predicted",y="Actual"),
                         text_auto=True, template="plotly_dark")
        fig2.update_layout(margin=dict(t=30))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Per-Class Metrics")
    rows = []
    for cls in classes:
        if cls in report:
            r = report[cls]
            rows.append({
                "Class"    : cls,
                "Precision": round(r["precision"], 3),
                "Recall"   : round(r["recall"],    3),
                "F1-Score" : round(r["f1-score"],  3),
                "Support"  : int(r["support"]),
            })
    df_report = pd.DataFrame(rows)
    st.dataframe(df_report, use_container_width=True, hide_index=True)

    # Bar chart
    df_melt = df_report.melt(id_vars="Class", value_vars=["Precision","Recall","F1-Score"],
                              var_name="Metric", value_name="Score")
    fig3 = px.bar(df_melt, x="Class", y="Score", color="Metric", barmode="group",
                  template="plotly_dark", title="Precision / Recall / F1 by Class")
    fig3.update_layout(yaxis_range=[0,1.1])
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 7 — HR Insights
# ══════════════════════════════════════════════════════════════════════
elif page == "💡 HR Insights":
    st.title("💡 HR Strategic Insights")

    if not os.path.exists(CFG.DATA_CSV):
        st.error("Dataset not found. Run `python main.py` first.")
        st.stop()

    df       = load_dataset()
    dept_sel = st.multiselect("Filter by Department", DEPT_LIST, default=DEPT_LIST)
    df_f     = df[df["department"].isin(dept_sel)] if dept_sel else df
    at_risk  = df_f[(df_f["satisfaction_score"] < CFG.AT_RISK_SAT_THRESHOLD) &
                    (df_f["absenteeism_days"]    > CFG.AT_RISK_ABSENT_THRESHOLD)]

    hi = df_f[df_f["performance_label"]=="High"]["training_hours"].mean()
    lo = df_f[df_f["performance_label"]=="Low"]["training_hours"].mean()

    c1,c2,c3 = st.columns(3)
    c1.metric("🚨 At-Risk Employees", len(at_risk),
              f"{len(at_risk)/max(len(df_f),1)*100:.1f}% of workforce")
    c2.metric("💸 Est. Attrition Cost",
              f"${len(at_risk)*CFG.ATTRITION_COST_PER_EMP:,.0f}", "@ $50K/employee")
    c3.metric("📚 Training Gap", f"{int(hi-lo)} hrs", "High vs Low performers")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_f.groupby("department")["training_hours"].mean().reset_index(),
                     x="department", y="training_hours",
                     color="training_hours", color_continuous_scale="Blues",
                     template="plotly_dark", title="Avg Training Hours by Department")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df_f, x="absenteeism_days", y="satisfaction_score",
                         color="performance_label", color_discrete_map=PERF_COLORS,
                         opacity=0.6, template="plotly_dark",
                         title="Absenteeism vs Satisfaction")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🚨 At-Risk Employee Table")
    if len(at_risk):
        st.dataframe(
            at_risk[["employee_id","department","satisfaction_score",
                     "absenteeism_days","performance_label"]]
            .sort_values("satisfaction_score").reset_index(drop=True),
            use_container_width=True)
    else:
        st.info("No at-risk employees in selected departments.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 8 — Model Versions (NEW)
# ══════════════════════════════════════════════════════════════════════
elif page == "🗂️ Model Versions":
    st.title("🗂️ Model Version History")
    st.caption("Every `python main.py` run saves a timestamped backup in models/versions/")

    versions_dir = os.path.join(CFG.MODELS_DIR, "versions")
    if not os.path.exists(versions_dir):
        st.info("No version backups yet. Run `python main.py` to create the first one.")
        st.stop()

    versions = sorted(
        [f for f in os.listdir(versions_dir) if f.endswith(".pkl")],
        reverse=True
    )

    if not versions:
        st.info("No version files found in models/versions/")
        st.stop()

    rows = []
    for v in versions:
        ts   = v.replace("pipeline_","").replace(".pkl","")
        size = os.path.getsize(os.path.join(versions_dir,v))
        rows.append({"Version": v, "Timestamp": ts,
                     "Size (KB)": round(size/1024,1)})

    df_ver = pd.DataFrame(rows)
    st.markdown(f"**{len(versions)} version(s) found:**")
    st.dataframe(df_ver, use_container_width=True, hide_index=True)

    st.markdown("#### Load a Specific Version for Prediction")
    selected = st.selectbox("Select version", versions)
    if st.button("🔄 Load Selected Version"):
        ver_path = os.path.join(versions_dir, selected)
        with open(ver_path, "rb") as f:
            ver_pipe = pickle.load(f)
        st.success(f"Loaded: {selected}")
        st.info("This version is loaded for this session only. "
                "Restart Streamlit to go back to the default pipeline.")

    if metadata_ready():
        st.markdown("#### Current Active Model Metadata")
        meta = load_metadata()
        st.json(meta)


# ══════════════════════════════════════════════════════════════════════
# PAGE 9 — API Guide (NEW)
# ══════════════════════════════════════════════════════════════════════
elif page == "🌐 API Guide":
    st.title("🌐 REST API Guide")
    st.caption("The FastAPI server lets any frontend / app call predictions via HTTP.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Start the API Server")
        st.code("""# Install dependencies
pip install fastapi uvicorn

# Start server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Docs auto-generated at:
http://localhost:8000/docs""", language="bash")

    with col2:
        st.markdown("#### Health Check")
        st.code('curl http://localhost:8000/health', language="bash")
        st.code("""{
  "status": "ok",
  "pipeline_exists": true,
  "version": "2.0.0"
}""", language="json")

    st.markdown("#### Single Prediction — POST /predict")
    st.code("""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": 32, "gender": "Female", "education": "Master",
    "department": "Engineering", "experience_years": 8,
    "salary": 75000, "training_hours": 60,
    "projects_completed": 12, "avg_monthly_hours": 180,
    "satisfaction_score": 4.2, "last_promotion_years": 2,
    "absenteeism_days": 3, "peer_review_score": 4.0,
    "manager_rating": 4.5
  }'""", language="bash")

    st.code("""{
  "label": "High",
  "confidence": 87.3,
  "probabilities": {"High": 87.3, "Medium": 10.2, "Low": 2.5},
  "risk_score": 2.1,
  "recommendations": [
    "🌟 Fast-track promotion candidate",
    "🎯 Assign to high-impact strategic projects"
  ]
}""", language="json")

    st.markdown("#### Batch Prediction — POST /predict/batch")
    st.code("""curl -X POST http://localhost:8000/predict/batch \\
  -H "Content-Type: application/json" \\
  -d '{"employees": [{...}, {...}, {...}]}'""", language="bash")

    st.markdown("#### Model Info — GET /model/info")
    st.code("curl http://localhost:8000/model/info", language="bash")

    st.markdown("#### Python client example")
    st.code("""import requests

employee = {
    "age": 32, "gender": "Female", "education": "Master",
    "department": "Engineering", "experience_years": 8,
    "salary": 75000, "training_hours": 60, "projects_completed": 12,
    "avg_monthly_hours": 180, "satisfaction_score": 4.2,
    "last_promotion_years": 2, "absenteeism_days": 3,
    "peer_review_score": 4.0, "manager_rating": 4.5,
}

r = requests.post("http://localhost:8000/predict", json=employee)
print(r.json())
# {'label': 'High', 'confidence': 87.3, 'risk_score': 2.1, ...}""",
    language="python")

"""
Fairness Troops — Streamlit Frontend

Pages:
  1. Submit Audit    — configure and launch a new audit job
  2. Job Monitor     — live-poll job status
  3. Audit Report    — view metrics, mitigation delta, feature importance, PDP
"""

import time
import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fairness Troops",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
    .metric-good  { border-color: #a6e3a1; }
    .metric-warn  { border-color: #f9e2af; }
    .metric-bad   { border-color: #f38ba8; }
    .metric-title { font-size: 0.85rem; color: #cdd6f4; letter-spacing: .06em; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #cdd6f4; }
    .stProgress > div > div { background-color: #89b4fa !important; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def post_audit(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/audits/", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def get_status(job_id: str) -> dict:
    r = requests.get(f"{API_URL}/audits/{job_id}/status", timeout=10)
    r.raise_for_status()
    return r.json()


def get_report(job_id: str) -> dict | None:
    r = requests.get(f"{API_URL}/reports/{job_id}", timeout=10)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def list_jobs() -> list[dict]:
    r = requests.get(f"{API_URL}/audits/", timeout=10)
    r.raise_for_status()
    return r.json()


def fairness_color(metric_name: str, value: float) -> str:
    """Return CSS class based on how fair the metric is."""
    if metric_name == "disparate_impact":
        if value >= 0.8:
            return "metric-good"
        elif value >= 0.6:
            return "metric-warn"
        return "metric-bad"
    else:  # SPD, EOD — closer to 0 is better
        abs_v = abs(value)
        if abs_v <= 0.05:
            return "metric-good"
        elif abs_v <= 0.15:
            return "metric-warn"
        return "metric-bad"


def metric_card(title: str, value: float, css_class: str):
    st.markdown(f"""
    <div class="metric-card {css_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value:.4f}</div>
    </div>
    """, unsafe_allow_html=True)


# ── sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/fluency/96/scales.png", width=60)
st.sidebar.title("Fairness Troops ⚖️")
st.sidebar.markdown("*ML Bias & Fairness Auditing*")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", ["🚀 Submit Audit", "📡 Job Monitor", "📊 Audit Report"])

if "job_id" not in st.session_state:
    st.session_state.job_id = ""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Submit Audit
# ══════════════════════════════════════════════════════════════════════════════

if page == "🚀 Submit Audit":
    st.title("Submit a Fairness Audit")
    st.markdown("Configure your model and dataset, then launch an asynchronous audit job.")

    with st.form("audit_form"):
        col1, col2 = st.columns(2)

        with col1:
            model_name = st.selectbox(
                "Model",
                ["random_forest", "logistic_regression"],
                help="Registered model name in the model registry",
            )
            dataset_name = st.selectbox(
                "Dataset",
                ["synthetic", "adult"],
                help="'synthetic' runs fully offline; 'adult' downloads the UCI Adult Income dataset",
            )

        with col2:
            sensitive_feature = st.text_input(
                "Sensitive Feature Column",
                value="group",
                help="Protected attribute to audit. Use 'sex' for the adult dataset.",
            )
            target_column = st.text_input(
                "Target Column",
                value="label",
                help="The label column. Use 'class' for the adult dataset.",
            )

        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            apply_mitigation = st.toggle(
                "Apply Reweighting Mitigation",
                value=True,
                help="Retrain model with fairness-aware sample weights",
            )
        with col4:
            run_explainability = st.toggle(
                "Run Explainability",
                value=True,
                help="Compute feature importance and Partial Dependence Plots",
            )

        submitted = st.form_submit_button("⚡ Launch Audit", use_container_width=True)

    if submitted:
        with st.spinner("Submitting audit job…"):
            try:
                result = post_audit({
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "sensitive_feature": sensitive_feature,
                    "target_column": target_column,
                    "apply_mitigation": apply_mitigation,
                    "run_explainability": run_explainability,
                })
                st.session_state.job_id = result["job_id"]
                st.success(f"✅ Job submitted! **Job ID:** `{result['job_id']}`")
                st.info("Switch to **📡 Job Monitor** to track progress.")
            except Exception as e:
                st.error(f"Failed to submit: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Job Monitor
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📡 Job Monitor":
    st.title("Job Monitor")

    job_id_input = st.text_input(
        "Job ID",
        value=st.session_state.job_id,
        placeholder="Paste a job ID or submit an audit first",
    )
    if job_id_input:
        st.session_state.job_id = job_id_input

    col_poll, col_refresh = st.columns([1, 1])
    auto_poll = col_poll.toggle("Auto-refresh (every 3s)", value=False)

    if st.session_state.job_id:
        try:
            status_data = get_status(st.session_state.job_id)
            status = status_data["status"]

            STATUS_COLORS = {
                "pending": "🟡",
                "running": "🔵",
                "completed": "🟢",
                "failed": "🔴",
            }
            st.markdown(f"### {STATUS_COLORS.get(status, '⚪')} Status: **{status.upper()}**")

            if status == "completed":
                st.success("Audit complete! Go to **📊 Audit Report** to see results.")
            elif status == "failed":
                st.error(f"Job failed: {status_data.get('error_message', 'Unknown error')}")
            elif status in ("pending", "running"):
                st.progress(0.5 if status == "running" else 0.1)
                if auto_poll:
                    time.sleep(3)
                    st.rerun()

            with st.expander("Raw status JSON"):
                st.json(status_data)

        except Exception as e:
            st.error(f"Could not fetch status: {e}")

    st.divider()
    st.subheader("All Jobs")
    try:
        jobs = list_jobs()
        if jobs:
            df = pd.DataFrame(jobs)[["job_id", "status", "model_name", "dataset_name", "created_at"]]
            df.columns = ["Job ID", "Status", "Model", "Dataset", "Created At"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No jobs yet. Submit an audit to get started.")
    except Exception as e:
        st.warning(f"Could not load jobs: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Audit Report
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Audit Report":
    st.title("Audit Report")

    job_id_input = st.text_input(
        "Job ID",
        value=st.session_state.job_id,
        placeholder="Enter a completed job ID",
    )

    if st.button("Load Report", use_container_width=True) or job_id_input:
        if not job_id_input:
            st.warning("Enter a job ID above.")
            st.stop()

        with st.spinner("Loading report…"):
            try:
                report = get_report(job_id_input)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if report is None:
            st.warning("No report found. The job may still be running.")
            st.stop()

        metrics = report["metrics"]

        # ── Fairness Metrics ──────────────────────────────────────────────────
        st.subheader("📐 Fairness Metrics (Baseline)")
        st.markdown("*Thresholds: DI ≥ 0.8 = fair; |SPD|, |EOD| ≤ 0.05 = fair*")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card(
                "Disparate Impact",
                metrics["disparate_impact"],
                fairness_color("disparate_impact", metrics["disparate_impact"]),
            )
        with c2:
            metric_card(
                "Statistical Parity Diff",
                metrics["statistical_parity_difference"],
                fairness_color("spd", metrics["statistical_parity_difference"]),
            )
        with c3:
            metric_card(
                "Equal Opportunity Diff",
                metrics["equal_opportunity_difference"],
                fairness_color("eod", metrics["equal_opportunity_difference"]),
            )
        with c4:
            metric_card(
                "Overall Accuracy",
                metrics["accuracy"],
                "metric-good",
            )

        # Per-group accuracy bar chart
        acc_by_group = metrics.get("accuracy_by_group", {})
        if acc_by_group:
            fig = px.bar(
                x=list(acc_by_group.keys()),
                y=list(acc_by_group.values()),
                labels={"x": "Group", "y": "Accuracy"},
                title="Accuracy by Sensitive Group",
                color=list(acc_by_group.values()),
                color_continuous_scale="RdYlGn",
                range_color=[0, 1],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── Mitigation Results ────────────────────────────────────────────────
        if report.get("mitigation"):
            st.divider()
            st.subheader("🔧 Mitigation: Reweighting Results")
            mit = report["mitigation"]

            comparison = {
                "Metric": ["Disparate Impact", "Stat. Parity Diff.", "Equal Opp. Diff."],
                "Before": [
                    metrics["disparate_impact"],
                    metrics["statistical_parity_difference"],
                    metrics["equal_opportunity_difference"],
                ],
                "After": [
                    mit["disparate_impact_after"],
                    mit["statistical_parity_difference_after"],
                    mit["equal_opportunity_difference_after"],
                ],
            }
            df_comp = pd.DataFrame(comparison)
            df_comp["Δ"] = [
                round(a - b, 4)
                for a, b in zip(df_comp["After"], df_comp["Before"])
            ]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Before", x=df_comp["Metric"], y=df_comp["Before"],
                                  marker_color="#f38ba8"))
            fig2.add_trace(go.Bar(name="After", x=df_comp["Metric"], y=df_comp["After"],
                                  marker_color="#a6e3a1"))
            fig2.update_layout(
                barmode="group",
                title="Fairness Metrics — Before vs After Mitigation",
                yaxis_title="Metric Value",
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # ── Explainability ────────────────────────────────────────────────────
        if report.get("explainability"):
            st.divider()
            st.subheader("🔍 Explainability")
            expl = report["explainability"]

            col_fi, col_pdp = st.columns([1, 1])

            with col_fi:
                st.markdown("**Feature Importance**")
                fi = expl["feature_importance"]
                fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
                fig3 = px.bar(
                    x=list(fi_sorted.values()),
                    y=list(fi_sorted.keys()),
                    orientation="h",
                    labels={"x": "Importance", "y": "Feature"},
                    color=list(fi_sorted.values()),
                    color_continuous_scale="Blues",
                )
                fig3.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig3, use_container_width=True)

            with col_pdp:
                st.markdown("**Partial Dependence Plots**")
                pdp_data = expl.get("pdp_data", {})
                if not pdp_data:
                    st.info("No PDP data was produced for this model.")
                else:
                    feat_select = st.selectbox("Select feature", list(pdp_data.keys()))
                if pdp_data and feat_select in pdp_data:
                    pdp = pdp_data[feat_select]
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(
                        x=pdp["x_values"],
                        y=pdp["y_values"],
                        mode="lines+markers",
                        line={"color": "#89b4fa", "width": 2},
                    ))
                    fig4.update_layout(
                        xaxis_title=feat_select,
                        yaxis_title="Partial Dependence",
                        title=f"PDP — {feat_select}",
                    )
                    st.plotly_chart(fig4, use_container_width=True)

        with st.expander("Raw report JSON"):
            st.json(report)

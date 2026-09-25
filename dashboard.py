"""
dashboard.py
----------------------------------
WEEK 3: Unified interactive dashboard combining:
- Week 1: risk prediction from complaint trends
- Week 2: complaint-to-risk traceability matching + emerging (unknown) risk detection

Run with:  streamlit run dashboard.py

This dashboard reads the CSV outputs already produced by
generate_synthetic_data.py, generate_fmea.py, risk_prediction.py, and
complaint_matching.py. Run those four scripts first if the outputs/
folder is empty.
"""

import os
import subprocess
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Medical Device Risk Intelligence Prototype", layout="wide")

# ---------------------------------------------------------------
# First-run setup: if the data/outputs files don't exist yet (e.g. on a
# fresh cloud deployment where only the .py files were uploaded), generate
# them automatically by running the pipeline scripts in order.
# ---------------------------------------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

REQUIRED_FILES = [
    "data/complaints.csv",
    "data/fmea_risk_items.csv",
    "outputs/risk_scores_full.csv",
    "outputs/complaint_matches.csv",
]

if not all(os.path.exists(f) for f in REQUIRED_FILES):
    with st.spinner("First-time setup: generating synthetic data and running models... (about 30-60 seconds)"):
        subprocess.run(["python", "generate_synthetic_data.py"], check=True)
        subprocess.run(["python", "generate_fmea.py"], check=True)
        subprocess.run(["python", "risk_prediction.py"], check=True)
        subprocess.run(["python", "complaint_matching.py"], check=True)

# ---------------------------------------------------------------
# Load data (cached so the app stays fast when you switch products)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    risk_scores = pd.read_csv("outputs/risk_scores_full.csv", parse_dates=["date"])
    matches = pd.read_csv("outputs/complaint_matches.csv")
    try:
        emerging = pd.read_csv("outputs/emerging_risk_clusters.csv")
    except FileNotFoundError:
        emerging = pd.DataFrame()
    return risk_scores, matches, emerging

risk_scores, matches, emerging = load_data()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("Medical Device Risk Intelligence — Prototype")
st.caption(
    "A 1-month student prototype demonstrating AI-assisted early risk detection and "
    "complaint-to-risk traceability. Built on fully synthetic data — proof of concept, "
    "not a validated production system."
)

products = sorted(risk_scores["product_id"].unique())
selected_product = st.selectbox("Select a device", products)

st.divider()

# ---------------------------------------------------------------
# Section 1: Risk score over time (Week 1)
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)

sub = risk_scores[risk_scores["product_id"] == selected_product].sort_values("date")
latest = sub.iloc[-1]

col1.metric("Current risk score", f"{latest['risk_score']:.2f}")
col2.metric("Complaints (last 30 rows)", int(sub["complaint_count"].tail(30).sum()))
col3.metric("High-severity (last 30 rows)", int(sub["high_severity_count"].tail(30).sum()))

st.subheader("Risk score vs. complaint volume over time")
fig, ax1 = plt.subplots(figsize=(11, 4))
ax1.plot(sub["date"], sub["risk_score"], color="crimson", linewidth=1.8, label="Predicted risk score")
ax1.set_ylabel("Risk score (0-1)", color="crimson")
ax1.set_ylim(0, 1)
ax2 = ax1.twinx()
ax2.bar(sub["date"], sub["complaint_count"], color="steelblue", alpha=0.3, width=1)
ax2.set_ylabel("Daily complaints", color="steelblue")
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ---------------------------------------------------------------
# Section 2: Complaint matching (Week 2)
# ---------------------------------------------------------------
st.subheader("Complaint \u2192 known risk matching")

prod_matches = matches[matches["product_id"] == selected_product]
matched_count = (prod_matches["status"] == "matched").sum()
unmatched_count = (prod_matches["status"] == "unmatched").sum()

mcol1, mcol2 = st.columns(2)
mcol1.metric("Matched to known risk", matched_count)
mcol2.metric("Unmatched (no known risk)", unmatched_count)

with st.expander("See matched complaints and their confidence scores"):
    st.dataframe(
        prod_matches[prod_matches["status"] == "matched"]
        [["complaint_id", "complaint_text", "best_match_risk_id", "best_match_text", "confidence"]]
        .sort_values("confidence", ascending=False),
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------------------
# Section 3: Emerging / unknown risk detection (the novel contribution)
# ---------------------------------------------------------------
st.subheader("\u26a0\ufe0f Emerging risks (novel: unmatched complaints that cluster together)")
st.caption(
    "These are complaints that did NOT match any documented risk item well, but closely "
    "resemble other unmatched complaints for this device \u2014 a possible sign of an "
    "undocumented failure mode."
)

prod_emerging = emerging[emerging["product_id"] == selected_product] if not emerging.empty else emerging

if prod_emerging.empty:
    st.info("No emerging risk clusters detected for this device with the current threshold.")
else:
    for _, row in prod_emerging.iterrows():
        st.warning(
            f"**{row['num_complaints']} similar unmatched complaints** found "
            f"(cluster: {row['cluster_id']})\n\n"
            f"Example: \"{row['sample_text']}\""
        )

st.divider()
st.caption(
    "Scope note: all data is synthetic, generated to demonstrate the AI/ML methodology. "
    "This prototype is not validated for real safety or regulatory decision-making."
)

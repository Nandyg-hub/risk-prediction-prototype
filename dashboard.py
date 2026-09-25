import os
import sys
import subprocess
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="MedRisk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL LIGHT THEME
# =========================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #f7f9fc;
}

[data-testid="stHeader"] {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */
h1 {
    color: #12355b !important;
}

h2 {
    color: #12355b !important;
}

h3 {
    color: #12355b !important;
}

/* Normal text */
p {
    color: #334155;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
}

[data-testid="stSidebar"] * {
    color: #26374a;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #dbe3ec;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #12355b !important;
}

/* Input labels */
label {
    color: #334155 !important;
}

/* Select boxes */
[data-baseweb="select"] {
    background-color: white !important;
}

/* Text inputs */
input,
textarea {
    background-color: white !important;
    color: #1e293b !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #7a869a;
    padding-top: 40px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FIRST TIME SETUP
# =========================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

REQUIRED_FILES = [
    "data/complaints.csv",
    "data/fmea_risk_items.csv",
    "outputs/risk_scores_full.csv",
    "outputs/complaint_matches.csv",
]

if not all(os.path.exists(file) for file in REQUIRED_FILES):

    with st.spinner(
        "Setting up MedRisk AI for the first time..."
    ):

        for script in [
            "generate_synthetic_data.py",
            "generate_fmea.py",
            "risk_prediction.py",
            "complaint_matching.py",
        ]:

            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:

                st.error(
                    f"Setup failed while running {script}"
                )

                st.code(result.stderr)

                st.stop()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    risk_scores = pd.read_csv(
        "outputs/risk_scores_full.csv",
        parse_dates=["date"]
    )

    matches = pd.read_csv(
        "outputs/complaint_matches.csv"
    )

    try:
        emerging = pd.read_csv(
            "outputs/emerging_risk_clusters.csv"
        )

    except FileNotFoundError:
        emerging = pd.DataFrame()

    return risk_scores, matches, emerging


risk_scores, matches, emerging = load_data()

products = sorted(
    risk_scores["product_id"].unique()
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏥 MedRisk AI")

st.sidebar.caption(
    "Medical Device Risk Intelligence"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📝 Report Complaint",
        "📊 Risk Dashboard",
        "⚠️ Emerging Risks",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "AI-assisted medical device risk monitoring"
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("MedRisk AI")

    st.subheader(
        "Medical Device Risk Intelligence Platform"
    )

    st.write(
        "Monitor medical device complaints, analyse risk "
        "trends and identify potential emerging safety "
        "concerns using AI-assisted analysis."
    )

    st.write("")

    if st.button(
        "📝 Start Risk Assessment",
        type="primary",
        use_container_width=True
    ):

        st.info(
            "Select 'Report Complaint' from the sidebar "
            "to enter a device complaint."
        )

    st.divider()

    st.subheader("How MedRisk AI Works")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 1️⃣ Collect")

        st.write(
            "Record medical device complaints, "
            "dates and severity."
        )

    with col2:

        st.markdown("### 2️⃣ Analyse")

        st.write(
            "Analyse complaint volume, severity "
            "and historical trends."
        )

    with col3:

        st.markdown("### 3️⃣ Predict")

        st.write(
            "Use machine-learning based analysis "
            "to estimate device risk."
        )

    st.divider()

    st.subheader("Current System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Devices Monitored",
            len(products)
        )

    with col2:

        st.metric(
            "Complaint Records",
            len(matches)
        )

    with col3:

        st.metric(
            "Risk Records",
            len(risk_scores)
        )

    st.info(
        "💡 Use the sidebar to report a complaint or "
        "explore the device risk dashboard."
    )


# =========================================================
# REPORT COMPLAINT
# =========================================================

elif page == "📝 Report Complaint":

    st.title("Report a Device Complaint")

    st.write(
        "Enter information about a medical device complaint."
    )

    st.divider()

    st.subheader("Device Information")

    selected_device = st.selectbox(
        "Medical Device",
        products
    )

    complaint_date = st.date_input(
        "Complaint Date"
    )

    st.subheader("Complaint Details")

    complaint_text = st.text_area(
        "Describe the complaint",
        placeholder=(
            "Example: Device stopped working during operation..."
        ),
        height=150
    )

    severity = st.slider(
        "Complaint Severity",
        min_value=1,
        max_value=5,
        value=3
    )

    severity_names = {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    }

    st.caption(
        f"Selected severity: "
        f"**{severity_names[severity]} ({severity}/5)**"
    )

    st.write("")

    analyse = st.button(
        "🔍 Analyse Complaint",
        type="primary",
        use_container_width=True
    )

    if analyse:

        if not complaint_text.strip():

            st.warning(
                "Please enter a complaint description first."
            )

        else:

            st.session_state["new_complaint"] = {
                "device": selected_device,
                "date": str(complaint_date),
                "complaint": complaint_text,
                "severity": severity
            }

            st.success(
                "Complaint captured successfully."
            )

            st.divider()

            st.subheader("Complaint Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Device",
                    selected_device
                )

            with col2:
                st.metric(
                    "Severity",
                    f"{severity}/5"
                )

            with col3:
                st.metric(
                    "Date",
                    str(complaint_date)
                )

            st.info(
                "The complaint has been captured. "
                "The next step is to connect this information "
                "to the machine-learning prediction pipeline."
            )


# =========================================================
# RISK DASHBOARD
# =========================================================

elif page == "📊 Risk Dashboard":

    st.title("Device Risk Dashboard")

    st.write(
        "Monitor complaint trends and predicted device risk."
    )

    selected_product = st.selectbox(
        "Select a medical device",
        products
    )

    sub = risk_scores[
        risk_scores["product_id"] == selected_product
    ].sort_values("date")

    latest = sub.iloc[-1]

    st.divider()

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Risk Score",
            f"{latest['risk_score']:.2f}"
        )

    with col2:

        st.metric(
            "Complaints",
            int(
                sub["complaint_count"]
                .tail(30)
                .sum()
            )
        )

    with col3:

        st.metric(
            "High Severity",
            int(
                sub["high_severity_count"]
                .tail(30)
                .sum()
            )
        )

    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    risk_score = float(
        latest["risk_score"]
    )

    if risk_score >= 0.70:

        risk_level = "HIGH"

        st.error(
            f"🔴 Current predicted risk: "
            f"**{risk_level} ({risk_score:.2f})**"
        )

    elif risk_score >= 0.40:

        risk_level = "MODERATE"

        st.warning(
            f"🟠 Current predicted risk: "
            f"**{risk_level} ({risk_score:.2f})**"
        )

    else:

        risk_level = "LOW"

        st.success(
            f"🟢 Current predicted risk: "
            f"**{risk_level} ({risk_score:.2f})**"
        )

    st.divider()

    # -----------------------------------------------------
    # GRAPH
    # -----------------------------------------------------

    st.subheader(
        "Risk Score vs Complaint Volume"
    )

    fig, ax1 = plt.subplots(
        figsize=(11, 4)
    )

    ax1.plot(
        sub["date"],
        sub["risk_score"],
        linewidth=2
    )

    ax1.set_ylabel(
        "Risk Score (0–1)"
    )

    ax1.set_ylim(
        0,
        1
    )

    ax2 = ax1.twinx()

    ax2.bar(
        sub["date"],
        sub["complaint_count"],
        alpha=0.25
    )

    ax2.set_ylabel(
        "Daily Complaints"
    )

    fig.tight_layout()

    st.pyplot(fig)

    st.divider()

    # -----------------------------------------------------
    # COMPLAINT MATCHING
    # -----------------------------------------------------

    st.subheader(
        "Complaint → Known Risk Matching"
    )

    prod_matches = matches[
        matches["product_id"] == selected_product
    ]

    matched_count = (
        prod_matches["status"] == "matched"
    ).sum()

    unmatched_count = (
        prod_matches["status"] == "unmatched"
    ).sum()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Matched to Known Risk",
            matched_count
        )

    with col2:

        st.metric(
            "Unmatched Complaints",
            unmatched_count
        )

    with st.expander(
        "View Matching Details"
    ):

        matched_data = prod_matches[
            prod_matches["status"] == "matched"
        ]

        if not matched_data.empty:

            st.dataframe(
                matched_data[
                    [
                        "complaint_id",
                        "complaint_text",
                        "best_match_risk_id",
                        "best_match_text",
                        "confidence"
                    ]
                ].sort_values(
                    "confidence",
                    ascending=False
                ),
                use_container_width=True
            )

        else:

            st.info(
                "No matched complaints available."
            )


# =========================================================
# EMERGING RISKS
# =========================================================

elif page == "⚠️ Emerging Risks":

    st.title("Emerging Risks")

    st.write(
        "Potential undocumented failure patterns identified "
        "from unmatched complaints."
    )

    selected_product = st.selectbox(
        "Select a medical device",
        products
    )

    st.divider()

    if emerging.empty:

        st.info(
            "No emerging risk data is available."
        )

    else:

        prod_emerging = emerging[
            emerging["product_id"] == selected_product
        ]

        if prod_emerging.empty:

            st.success(
                "No emerging risk clusters detected "
                "for this device."
            )

        else:

            st.subheader(
                "Potential Emerging Risk Patterns"
            )

            for _, row in prod_emerging.iterrows():

                st.warning(
                    f"**{row['num_complaints']} similar "
                    f"unmatched complaints detected**"
                )

                st.write(
                    f"Example complaint: "
                    f"“{row['sample_text']}”"
                )

                st.caption(
                    f"Cluster ID: {row['cluster_id']}"
                )

                st.divider()


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.title("About MedRisk AI")

    st.subheader(
        "Medical Device Risk Intelligence"
    )

    st.write(
        "MedRisk AI is a research prototype designed "
        "to analyse medical device complaint patterns "
        "and identify potential emerging safety risks."
    )

    st.divider()

    st.subheader("Key Features")

    st.markdown("""
    - 📊 Complaint trend analysis
    - 🤖 Machine-learning based risk prediction
    - 🔗 Complaint-to-risk matching
    - ⚠️ Emerging risk detection
    - 📈 Device-level risk monitoring
    """)

    st.divider()

    st.subheader("Technology")

    st.write(
        "Python • Pandas • Scikit-learn • Streamlit"
    )

    st.divider()

    st.warning(
        "⚠️ This is a research/prototype system. "
        "The current dataset is synthetic and the system "
        "is not validated for clinical or regulatory "
        "decision-making."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'MedRisk AI · Medical Device Risk Intelligence<br>'
    'AI-assisted research prototype'
    '</div>',
    unsafe_allow_html=True
)

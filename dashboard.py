import os
import sys
import subprocess
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedRisk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #12355b;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #5f6b7a;
        margin-bottom: 25px;
    }

    /* Cards */
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e5e9f0;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 20px;
        font-weight: 600;
        color: #12355b;
        margin-bottom: 8px;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #eaf4ff, #ffffff);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid #dcecff;
        margin-bottom: 30px;
    }

    .hero h1 {
        color: #12355b;
        font-size: 40px;
    }

    .hero p {
        color: #536273;
        font-size: 18px;
        line-height: 1.6;
    }

    /* Risk box */
    .risk-box {
        background-color: white;
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #e5e9f0;
        text-align: center;
        margin: 15px 0;
    }

    .risk-number {
        font-size: 42px;
        font-weight: 700;
        color: #12355b;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7b8794;
        font-size: 13px;
        padding: 30px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FIRST RUN SETUP
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

REQUIRED_FILES = [
    "data/complaints.csv",
    "data/fmea_risk_items.csv",
    "outputs/risk_scores_full.csv",
    "outputs/complaint_matches.csv",
]

if not all(os.path.exists(f) for f in REQUIRED_FILES):

    with st.spinner(
        "Setting up the risk prediction system for the first time..."
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

                st.error(f"Setup failed while running {script}")

                st.code(result.stderr)

                st.stop()


# ============================================================
# LOAD DATA
# ============================================================

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


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("## 🏥 MedRisk AI")

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


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">

        <div class="main-title">
            MedRisk AI
        </div>

        <div class="subtitle">
            Medical Device Risk Intelligence Platform
        </div>

        <p>
            Monitor medical device complaints, analyse risk trends,
            and identify potential emerging safety concerns using
            AI-assisted analysis.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("How it works")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        1️⃣ Collect
        </div>

        Record medical device complaints and
        their severity.

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        2️⃣ Analyse
        </div>

        Analyse complaint volume, severity
        and historical trends.

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        3️⃣ Predict
        </div>

        Use machine-learning based risk
        prediction to identify potential
        future risk.

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Current System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Devices monitored",
            len(products)
        )

    with col2:
        st.metric(
            "Complaint records",
            len(matches)
        )

    with col3:
        st.metric(
            "Risk records",
            len(risk_scores)
        )

    st.divider()

    st.info(
        "💡 Start by selecting 'Report Complaint' from the sidebar."
    )


# ============================================================
# REPORT COMPLAINT PAGE
# ============================================================

elif page == "📝 Report Complaint":

    st.markdown(
        '<div class="main-title">Report a Device Complaint</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter information about a medical device complaint.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

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
        height=140
    )

    severity = st.slider(
        "Complaint Severity",
        min_value=1,
        max_value=5,
        value=3
    )

    severity_text = {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    }

    st.caption(
        f"Selected severity: **{severity_text[severity]}**"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(
        "🔍 Analyse Complaint",
        use_container_width=True
    ):

        if not complaint_text.strip():

            st.warning(
                "Please enter a complaint description."
            )

        else:

            st.session_state["new_complaint"] = {
                "device": selected_device,
                "date": str(complaint_date),
                "complaint": complaint_text,
                "severity": severity
            }

            st.success(
                "Complaint information captured successfully."
            )

            st.markdown("### Complaint Summary")

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
                "The next development step will connect this "
                "new complaint directly to the machine-learning "
                "risk prediction pipeline."
            )


# ============================================================
# RISK DASHBOARD
# ============================================================

elif page == "📊 Risk Dashboard":

    st.markdown(
        '<div class="main-title">Risk Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor device-level complaint trends and predicted risk.'
        '</div>',
        unsafe_allow_html=True
    )

    selected_product = st.selectbox(
        "Select a medical device",
        products
    )

    st.divider()

    sub = risk_scores[
        risk_scores["product_id"] == selected_product
    ].sort_values("date")

    latest = sub.iloc[-1]

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

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

    st.divider()

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    risk_score = float(
        latest["risk_score"]
    )

    if risk_score >= 0.70:
        risk_level = "HIGH"
    elif risk_score >= 0.40:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    st.markdown(
        f"""
        <div class="risk-box">

            <div style="font-size:18px;color:#667085;">
                Current predicted risk
            </div>

            <div class="risk-number">
                {risk_score:.2f}
            </div>

            <div style="font-size:24px;font-weight:600;">
                {risk_level} RISK
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    st.subheader(
        "Risk Score vs Complaint Volume"
    )

    fig, ax1 = plt.subplots(
        figsize=(11, 4)
    )

    ax1.plot(
        sub["date"],
        sub["risk_score"],
        linewidth=2,
        label="Predicted risk"
    )

    ax1.set_ylabel(
        "Risk score (0–1)"
    )

    ax1.set_ylim(
        0,
        1
    )

    ax2 = ax1.twinx()

    ax2.bar(
        sub["date"],
        sub["complaint_count"],
        alpha=0.25,
        width=1
    )

    ax2.set_ylabel(
        "Daily complaints"
    )

    fig.tight_layout()

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # COMPLAINT MATCHING
    # --------------------------------------------------------

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
        "View complaint matching details"
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


# ============================================================
# EMERGING RISKS
# ============================================================

elif page == "⚠️ Emerging Risks":

    st.markdown(
        '<div class="main-title">Emerging Risks</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Potential undocumented failure patterns identified '
        'from unmatched complaints.'
        '</div>',
        unsafe_allow_html=True
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
                "for this device with the current threshold."
            )

        else:

            st.subheader(
                "Potential Emerging Risk Patterns"
            )

            for _, row in prod_emerging.iterrows():

                st.warning(
                    f"**{row['num_complaints']} similar "
                    f"unmatched complaints** detected\n\n"
                    f"Example complaint: "
                    f"\"{row['sample_text']}\""
                )

                st.caption(
                    f"Cluster ID: {row['cluster_id']}"
                )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">About MedRisk AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-assisted medical device risk intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### What is MedRisk AI?

    MedRisk AI is a prototype platform designed to analyse
    medical device complaint patterns and identify potential
    emerging safety risks.

    ### Main capabilities

    - Complaint trend analysis
    - Machine-learning based risk prediction
    - Complaint-to-risk matching
    - Emerging risk detection
    - Device-level risk monitoring

    ### Technology

    **Python + Pandas + Scikit-learn + Streamlit**

    The current machine-learning pipeline uses a
    **Random Forest classifier** based on complaint-related
    features and historical trends.
    """)

    st.warning(
        "⚠️ This is a research/prototype system. "
        "The current dataset is synthetic and the system "
        "is not validated for real clinical or regulatory "
        "decision-making."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        MedRisk AI · Medical Device Risk Intelligence<br>
        AI-assisted prototype for research and demonstration
    </div>
    """,
    unsafe_allow_html=True
)

import os
import sys
import subprocess
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MedRisk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LIGHT THEME + UI STYLING
# ============================================================

st.markdown("""
<style>

/* ---------- MAIN PAGE ---------- */
.stApp {
    background-color: #f7f9fc;
    color: #12395b;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- HEADINGS ---------- */
h1, h2, h3, h4 {
    color: #123f68 !important;
}

p, label, span, div {
    color: #234b6d;
}

/* ---------- TOP NAV ---------- */
.navbar {
    background: white;
    border-bottom: 1px solid #d9e3ee;
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 2px 8px rgba(30, 70, 100, 0.06);
}

.brand {
    font-size: 28px;
    font-weight: 800;
    color: #0e3b63 !important;
}

.brand-sub {
    color: #66809a !important;
    font-size: 13px;
}

/* ---------- CARDS ---------- */
.card {
    background: white;
    border: 1px solid #dce6f0;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 3px 12px rgba(30, 70, 100, 0.06);
    margin-bottom: 18px;
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #12436c !important;
}

.card-value {
    font-size: 30px;
    font-weight: 700;
    color: #174e7d !important;
}

/* ---------- INFO BOX ---------- */
.info-box {
    background: #eaf4ff;
    border-left: 5px solid #3182ce;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 15px 0;
}

/* ---------- RISK BOXES ---------- */
.high-risk {
    background: #ffe4e4;
    border: 1px solid #ff9b9b;
    border-radius: 12px;
    padding: 18px;
    color: #a80000 !important;
    font-weight: 700;
}

.moderate-risk {
    background: #fff8d9;
    border: 1px solid #ead36b;
    border-radius: 12px;
    padding: 18px;
    color: #806400 !important;
    font-weight: 700;
}

.low-risk {
    background: #e3f7eb;
    border: 1px solid #8bd3a7;
    border-radius: 12px;
    padding: 18px;
    color: #16733b !important;
    font-weight: 700;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    border: none;
    background: #ff4b4b;
    color: white !important;
    font-weight: 600;
    padding: 0.65rem 1rem;
}

.stButton > button:hover {
    background: #e63e3e;
    color: white !important;
}

/* ---------- INPUTS ---------- */
div[data-baseweb="select"] > div,
textarea,
input {
    border-radius: 8px !important;
}

/* ---------- METRICS ---------- */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce6f0;
    border-radius: 12px;
    padding: 15px;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #dce6f0;
}

section[data-testid="stSidebar"] * {
    color: #234b6d !important;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    color: #8193a5 !important;
    font-size: 12px;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# FOLDER SETUP
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================
# GENERATE REQUIRED FILES IF MISSING
# ============================================================

required_files = [
    "data/complaints.csv",
    "data/fmea_risk_items.csv",
    "outputs/risk_scores_full.csv",
    "outputs/complaint_matches.csv"
]

if not all(os.path.exists(x) for x in required_files):

    scripts = [
        "generate_synthetic_data.py",
        "generate_fmea.py",
        "risk_prediction.py",
        "complaint_matching.py"
    ]

    with st.spinner("Preparing MedRisk AI model..."):

        for script in scripts:

            if not os.path.exists(script):
                st.error(f"Required file missing: {script}")
                st.stop()

            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error(f"Error while running {script}")
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

    complaints = pd.read_csv(
        "data/complaints.csv"
    )

    matches = pd.read_csv(
        "outputs/complaint_matches.csv"
    )

    try:
        fmea = pd.read_csv("data/fmea_risk_items.csv")
    except Exception:
        fmea = pd.DataFrame()

    try:
        emerging = pd.read_csv(
            "outputs/emerging_risk_clusters.csv"
        )
    except Exception:
        emerging = pd.DataFrame()

    return risk_scores, complaints, matches, fmea, emerging


risk_scores, complaints, matches, fmea, emerging = load_data()

products = sorted(
    risk_scores["product_id"].dropna().unique()
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:10px 5px 20px 5px;">
            <div style="font-size:24px;font-weight:800;color:#123f68;">
                🏥 MedRisk AI
            </div>
            <div style="font-size:12px;color:#71869a;">
                Medical Device Risk Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📝 Report Complaint",
            "📊 Risk Dashboard",
            "⚠️ Emerging Risks",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.caption("AI-assisted medical device risk monitoring")

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="navbar">
        <div class="brand">🏥 MedRisk AI</div>
        <div class="brand-sub">
            Medical Device Risk Intelligence Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_level(score):

    if score >= 0.70:
        return "HIGH RISK", "high-risk"

    elif score >= 0.40:
        return "MODERATE RISK", "moderate-risk"

    return "LOW RISK", "low-risk"


def keyword_score(text):

    text = text.lower()

    high_words = [
        "heating",
        "overheating",
        "smoke",
        "fire",
        "burn",
        "electric shock",
        "shock",
        "explosion",
        "sparking",
        "failed during operation",
        "stopped during operation",
        "unexpected shutdown",
        "wrong reading",
        "inaccurate reading"
    ]

    medium_words = [
        "failure",
        "failed",
        "error",
        "malfunction",
        "crack",
        "leak",
        "drift",
        "unstable",
        "intermittent",
        "delay"
    ]

    score = 0.0

    for word in high_words:
        if word in text:
            score += 0.35

    for word in medium_words:
        if word in text:
            score += 0.15

    return min(score, 0.60)


def text_fmea_match(description, product_id):

    if fmea.empty:
        return 0.0, None, None

    product_fmea = fmea[
        fmea["product_id"].astype(str) == str(product_id)
    ]

    if product_fmea.empty:
        return 0.0, None, None

    try:

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [description] + \
                product_fmea["description"].fillna("").tolist()

        vectorizer = TfidfVectorizer(stop_words="english")

        matrix = vectorizer.fit_transform(texts)

        similarities = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        )[0]

        best_index = int(np.argmax(similarities))

        score = float(similarities[best_index])

        risk_id = product_fmea.iloc[best_index].get(
            "risk_id",
            "Known risk"
        )

        risk_text = product_fmea.iloc[best_index].get(
            "description",
            ""
        )

        return score, risk_id, risk_text

    except Exception:
        return 0.0, None, None


def calculate_live_risk(
    product_id,
    severity,
    category,
    description
):

    # --------------------------------------------------------
    # Historical ML risk
    # --------------------------------------------------------

    product_history = risk_scores[
        risk_scores["product_id"] == product_id
    ].sort_values("date")

    if product_history.empty:
        historical_risk = 0.20
    else:
        historical_risk = float(
            product_history.iloc[-1]["risk_score"]
        )

    # --------------------------------------------------------
    # Severity contribution
    # --------------------------------------------------------

    severity_component = severity / 5.0

    # --------------------------------------------------------
    # Category contribution
    # --------------------------------------------------------

    category_weights = {
        "Battery failure": 0.80,
        "Software error": 0.65,
        "Mechanical defect": 0.65,
        "Labeling issue": 0.35,
        "Connector failure": 0.60,
        "Sensor inaccuracy": 0.70,
        "User error": 0.30,
        "Packaging defect": 0.55
    }

    category_component = category_weights.get(
        category,
        0.50
    )

    # --------------------------------------------------------
    # Complaint text analysis
    # --------------------------------------------------------

    text_component = keyword_score(description)

    # --------------------------------------------------------
    # FMEA similarity
    # --------------------------------------------------------

    fmea_score, risk_id, risk_text = text_fmea_match(
        description,
        product_id
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    score = (
        0.30 * historical_risk
        + 0.30 * severity_component
        + 0.15 * category_component
        + 0.15 * text_component
        + 0.10 * fmea_score
    )

    # Strong safety signals should push high-severity
    # complaints into the high-risk zone for demonstration.
    if severity >= 5 and text_component >= 0.30:
        score = max(score, 0.85)

    elif severity >= 4 and text_component >= 0.30:
        score = max(score, 0.70)

    score = min(max(score, 0.01), 0.98)

    return score, fmea_score, risk_id, risk_text


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("MedRisk AI")

    st.subheader(
        "Medical Device Risk Intelligence Platform"
    )

    st.write(
        "Analyse medical device complaints, identify known "
        "failure modes and detect potential emerging safety risks."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">1️⃣ Collect</div>
                <p>Enter device complaints, dates, category and severity.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">2️⃣ Analyse</div>
                <p>Compare complaints with known FMEA risks and historical trends.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">3️⃣ Predict</div>
                <p>Use machine-learning assisted analysis to estimate device risk.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Current System Overview")

    total_devices = len(products)
    total_complaints = len(complaints)

    if not fmea.empty:
        total_fmea = len(fmea)
    else:
        total_fmea = 0

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Devices Monitored",
            total_devices
        )

    with b:
        st.metric(
            "Complaint Records",
            total_complaints
        )

    with c:
        st.metric(
            "Known FMEA Risks",
            total_fmea
        )

    st.markdown(
        """
        <div class="info-box">
        💡 Use <b>Report Complaint</b> to enter a new device complaint
        and generate a live AI-assisted risk assessment.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# REPORT COMPLAINT
# ============================================================

elif page == "📝 Report Complaint":

    st.title("Report a Device Complaint")

    st.write(
        "Enter information about a medical device complaint "
        "to generate an AI-assisted risk assessment."
    )

    st.markdown("---")

    st.subheader("Device Information")

    selected_device = st.selectbox(
        "Medical Device",
        products
    )

    complaint_date = st.date_input(
        "Complaint Date",
        value=date.today()
    )

    st.subheader("Complaint Details")

    categories = [
        "Battery failure",
        "Software error",
        "Mechanical defect",
        "Labeling issue",
        "Connector failure",
        "Sensor inaccuracy",
        "User error",
        "Packaging defect"
    ]

    category = st.selectbox(
        "Failure / Complaint Category",
        categories
    )

    severity = st.slider(
        "Complaint Severity",
        min_value=1,
        max_value=5,
        value=3
    )

    severity_names = {
        1: "Minor",
        2: "Low",
        3: "Moderate",
        4: "Serious",
        5: "Critical"
    }

    st.caption(
        f"Selected severity: "
        f"**{severity_names[severity]} ({severity}/5)**"
    )

    description = st.text_area(
        "Describe the complaint",
        placeholder=(
            "Example: Battery is heating during use and "
            "the device shuts down unexpectedly."
        ),
        height=130
    )

    st.markdown("<br>", unsafe_allow_html=True)

    assess = st.button(
        "🔍 Assess Device Risk",
        use_container_width=True
    )

    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    if assess:

        if not description.strip():

            st.warning(
                "Please enter a complaint description before assessment."
            )

        else:

            score, fmea_score, risk_id, risk_text = calculate_live_risk(
                selected_device,
                severity,
                category,
                description
            )

            level, css_class = get_risk_level(score)

            st.markdown("---")

            st.subheader("🤖 AI-Assisted Risk Assessment")

            if level == "HIGH RISK":

                st.markdown(
                    f"""
                    <div class="high-risk">
                    🔴 HIGH RISK — Immediate review recommended.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif level == "MODERATE RISK":

                st.markdown(
                    f"""
                    <div class="moderate-risk">
                    🟠 MODERATE RISK — Further monitoring is recommended.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="low-risk">
                    🟢 LOW RISK — Continue routine monitoring.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "Risk Score",
                    f"{score:.2f}"
                )

            with r2:
                st.metric(
                    "Risk Probability",
                    f"{score * 100:.1f}%"
                )

            with r3:
                st.metric(
                    "Severity",
                    f"{severity}/5"
                )

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("Risk Analysis")

            reasons = []

            if severity >= 4:
                reasons.append(
                    f"High complaint severity ({severity}/5)"
                )

            if keyword_score(description) > 0:
                reasons.append(
                    "Safety-related terms detected in complaint description"
                )

            if fmea_score >= 0.35:
                reasons.append(
                    f"Complaint shows similarity to a documented FMEA risk"
                )

            if not reasons:
                reasons.append(
                    "Risk estimate is mainly influenced by historical device trends"
                )

            for reason in reasons:
                st.write("• " + reason)

            # ------------------------------------------------
            # Known FMEA Match
            # ------------------------------------------------

            if risk_id is not None:

                st.markdown("<br>", unsafe_allow_html=True)

                st.subheader("Known Risk Traceability")

                st.info(
                    f"**Possible FMEA match:** {risk_id}\n\n"
                    f"**Similarity:** {fmea_score:.2f}\n\n"
                    f"**Risk description:** {risk_text}"
                )

            else:

                st.info(
                    "No strong match was found against the documented "
                    "FMEA risks. The complaint can be monitored as a "
                    "potential undocumented failure mode."
                )

            # ------------------------------------------------
            # Action
            # ------------------------------------------------

            if level == "HIGH RISK":

                st.error(
                    "Recommended prototype action: "
                    "Flag complaint for immediate engineering / safety review."
                )

            elif level == "MODERATE RISK":

                st.warning(
                    "Recommended prototype action: "
                    "Continue monitoring and investigate recurring patterns."
                )

            else:

                st.success(
                    "Recommended prototype action: "
                    "Continue routine monitoring."
                )

# ============================================================
# RISK DASHBOARD
# ============================================================

elif page == "📊 Risk Dashboard":

    st.title("Device Risk Dashboard")

    st.write(
        "Monitor historical complaint trends and predicted device risk."
    )

    selected_product = st.selectbox(
        "Select a medical device",
        products,
        key="dashboard_product"
    )

    sub = risk_scores[
        risk_scores["product_id"] == selected_product
    ].sort_values("date")

    latest = sub.iloc[-1]

    current_risk = float(
        latest["risk_score"]
    )

    complaint_count = int(
        sub["complaint_count"].tail(30).sum()
    )

    high_severity = int(
        sub["high_severity_count"].tail(30).sum()
    )

    st.markdown("---")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Current Risk Score",
            f"{current_risk:.2f}"
        )

    with d2:
        st.metric(
            "Complaints",
            complaint_count
        )

    with d3:
        st.metric(
            "High Severity",
            high_severity
        )

    level, css_class = get_risk_level(
        current_risk
    )

    if level == "HIGH RISK":

        st.markdown(
            """
            <div class="high-risk">
            🔴 Current predicted risk: HIGH
            </div>
            """,
            unsafe_allow_html=True
        )

    elif level == "MODERATE RISK":

        st.markdown(
            """
            <div class="moderate-risk">
            🟠 Current predicted risk: MODERATE
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="low-risk">
            🟢 Current predicted risk: LOW
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader(
        "Risk Score vs Complaint Volume"
    )

    chart_data = sub[
        ["date", "risk_score", "complaint_count"]
    ].set_index("date")

    st.line_chart(
        chart_data[
            ["risk_score"]
        ]
    )

    st.bar_chart(
        chart_data[
            ["complaint_count"]
        ]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader(
        "Recent Device History"
    )

    recent = sub.tail(20).copy()

    recent["risk_score"] = recent[
        "risk_score"
    ].round(3)

    st.dataframe(
        recent[
            [
                "date",
                "complaint_count",
                "high_severity_count",
                "risk_score"
            ]
        ].sort_values(
            "date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# EMERGING RISKS
# ============================================================

elif page == "⚠️ Emerging Risks":

    st.title("Emerging / Undocumented Risks")

    st.write(
        "Complaints that do not strongly match documented FMEA risks "
        "are grouped to identify possible emerging failure patterns."
    )

    st.markdown("---")

    if emerging.empty:

        st.info(
            "No emerging risk clusters are currently available."
        )

    else:

        for product_id in products:

            prod = emerging[
                emerging["product_id"] == product_id
            ]

            if prod.empty:
                continue

            st.subheader(
                f"Device: {product_id}"
            )

            for _, row in prod.iterrows():

                st.warning(
                    f"""
                    **{row['num_complaints']} similar complaints detected**

                    Cluster: {row['cluster_id']}

                    Example complaint:

                    "{row['sample_text']}"
                    """
                )

# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("About MedRisk AI")

    st.subheader(
        "Medical Device Risk Intelligence Prototype"
    )

    st.write(
        """
        MedRisk AI is a student research prototype designed to
        demonstrate how complaint data can be combined with
        machine-learning based risk prediction and FMEA traceability.
        """
    )

    st.markdown("---")

    st.subheader("Core Pipeline")

    st.write(
        """
        **1. Complaint Collection**

        Medical device complaint information is entered into the system.

        **2. Complaint Analysis**

        Complaint category, severity, historical trends and text are analysed.

        **3. FMEA Traceability**

        The complaint is compared with documented failure modes.

        **4. Risk Prediction**

        Historical complaint trends are used by the prototype ML model
        to estimate future risk.

        **5. Emerging Risk Detection**

        Unmatched complaints that resemble each other can be grouped
        as possible undocumented failure patterns.
        """
    )

    st.markdown("---")

    st.info(
        "Prototype limitation: the current dataset is synthetic and "
        "the system has not been validated for clinical, safety or "
        "regulatory decision-making."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        MedRisk AI • Medical Device Risk Intelligence Prototype<br>
        Synthetic data • AI-assisted analysis • Proof of concept
    </div>
    """,
    unsafe_allow_html=True
)

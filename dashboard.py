"""
MedRisk AI
Medical Device Risk Intelligence Platform
Final student prototype dashboard

Features:
1. Home page
2. Live complaint assessment
3. Risk prediction
4. Complaint category + severity analysis
5. Complaint text risk analysis
6. Possible complication explanation
7. Risk Dashboard
8. Emerging Risks
9. About page

NOTE:
This is a student prototype using synthetic/demo data.
It is NOT a clinically or regulatorily validated system.
"""

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
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page spacing */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Prevent top content from getting hidden */
    header {
        visibility: visible;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #123f67;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 20px;
        font-weight: 600;
        color: #244f73;
        margin-bottom: 8px;
    }

    .description {
        font-size: 15px;
        color: #526779;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #d8e2ec;
        background: white;
        min-height: 125px;
    }

    .card-title {
        font-size: 14px;
        color: #587087;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 30px;
        font-weight: 700;
        color: #173f62;
    }

    /* Risk boxes */
    .high-risk {
        padding: 20px;
        border-radius: 12px;
        background: #ffe4e4;
        border: 2px solid #ff6b6b;
        color: #a40000;
        font-weight: 700;
        font-size: 18px;
    }

    .moderate-risk {
        padding: 20px;
        border-radius: 12px;
        background: #fff8cc;
        border: 2px solid #f0c84b;
        color: #805f00;
        font-weight: 700;
        font-size: 18px;
    }

    .low-risk {
        padding: 20px;
        border-radius: 12px;
        background: #e3f7e9;
        border: 2px solid #62c985;
        color: #146b35;
        font-weight: 700;
        font-size: 18px;
    }

    /* Section headings */
    .section-heading {
        color: #123f67;
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        padding-top: 1.5rem;
    }

    /* Hide unnecessary Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# OPTIONAL DATA SETUP
# ============================================================

def run_script(script_name):

    if not os.path.exists(script_name):
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True
        )

        return result.returncode == 0

    except Exception:
        return False


# Generate demo data only if complaint data is missing
if not os.path.exists("data/complaints.csv"):

    run_script("generate_synthetic_data.py")


# Generate FMEA data if possible
if not os.path.exists("data/fmea_risk_items.csv"):

    run_script("generate_fmea.py")


# Generate model outputs if possible
if not os.path.exists("outputs/risk_scores_full.csv"):

    run_script("risk_prediction.py")


if not os.path.exists("outputs/complaint_matches.csv"):

    run_script("complaint_matching.py")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    complaints = pd.DataFrame()
    risk_scores = pd.DataFrame()
    matches = pd.DataFrame()
    emerging = pd.DataFrame()

    try:
        complaints = pd.read_csv("data/complaints.csv")
    except Exception:
        pass

    try:
        risk_scores = pd.read_csv(
            "outputs/risk_scores_full.csv",
            parse_dates=["date"]
        )
    except Exception:
        pass

    try:
        matches = pd.read_csv(
            "outputs/complaint_matches.csv"
        )
    except Exception:
        pass

    try:
        emerging = pd.read_csv(
            "outputs/emerging_risk_clusters.csv"
        )
    except Exception:
        pass

    return complaints, risk_scores, matches, emerging


complaints, risk_scores, matches, emerging = load_data()


# ============================================================
# DEVICE LIST
# ============================================================

if not complaints.empty and "product_id" in complaints.columns:

    devices = sorted(
        complaints["product_id"].dropna().unique().tolist()
    )

else:

    devices = [
        f"DEV-{i:03d}"
        for i in range(1, 13)
    ]


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
        font-size:28px;
        font-weight:800;
        color:#123f67;
        margin-bottom:4px;">
        🏥 MedRisk AI
        </div>

        <div style="
        font-size:13px;
        color:#66788a;
        margin-bottom:25px;">
        Medical Device Risk Intelligence
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
        ],
        index=0
    )

    st.markdown("---")

    st.caption(
        "AI-assisted medical device risk monitoring"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_risk(score):

    if score >= 0.70:
        return "HIGH"

    elif score >= 0.40:
        return "MODERATE"

    return "LOW"


def get_risk_class(level):

    if level == "HIGH":
        return "high-risk"

    elif level == "MODERATE":
        return "moderate-risk"

    return "low-risk"


def analyse_complaint(
    device,
    category,
    severity,
    description
):

    """
    Live risk assessment layer.

    The historical ML score provides the baseline.
    Immediate complaint characteristics then modify
    the score so that serious real-time complaints
    are not hidden by the historical average.
    """

    # --------------------------------------------------------
    # 1. Historical baseline
    # --------------------------------------------------------

    baseline = 0.20

    if not risk_scores.empty:

        try:

            device_data = risk_scores[
                risk_scores["product_id"] == device
            ]

            if not device_data.empty:

                baseline = float(
                    device_data["risk_score"].iloc[-1]
                )

        except Exception:
            baseline = 0.20


    # --------------------------------------------------------
    # 2. Severity contribution
    # --------------------------------------------------------

    severity_score = severity / 5.0

    # --------------------------------------------------------
    # 3. Category contribution
    # --------------------------------------------------------

    category_weights = {

        "Battery failure": 0.85,

        "Software error": 0.60,

        "Mechanical defect": 0.70,

        "Labeling issue": 0.40,

        "Connector failure": 0.65,

        "Sensor inaccuracy": 0.75,

        "User error": 0.35,

        "Packaging defect": 0.65
    }

    category_score = category_weights.get(
        category,
        0.50
    )


    # --------------------------------------------------------
    # 4. Text-based warning signals
    # --------------------------------------------------------

    text = description.lower()

    critical_words = [
        "heating",
        "overheating",
        "burn",
        "smoke",
        "fire",
        "shock",
        "electric shock",
        "explosion",
        "explode",
        "sparking",
        "spark",
        "shutdown",
        "stopped working",
        "failed during operation",
        "injury",
        "patient injury",
        "error during treatment",
        "incorrect reading",
        "wrong reading",
        "life threatening",
        "life-threatening"
    ]

    warning_words = [
        "leak",
        "crack",
        "damage",
        "unstable",
        "intermittent",
        "malfunction",
        "failure",
        "inaccurate",
        "unexpected",
        "drift"
    ]

    critical_hits = [
        word for word in critical_words
        if word in text
    ]

    warning_hits = [
        word for word in warning_words
        if word in text
    ]


    # --------------------------------------------------------
    # 5. Recent complaint volume
    # --------------------------------------------------------

    recent_factor = 0.0

    if not complaints.empty:

        try:

            device_complaints = complaints[
                complaints["product_id"] == device
            ]

            recent_count = len(
                device_complaints.tail(30)
            )

            if recent_count >= 15:
                recent_factor = 0.10

            elif recent_count >= 8:
                recent_factor = 0.05

        except Exception:
            recent_factor = 0.0


    # --------------------------------------------------------
    # 6. Calculate live risk score
    # --------------------------------------------------------

    score = (

        0.25 * baseline
        + 0.30 * severity_score
        + 0.20 * category_score
        + recent_factor
    )


    # Warning terms
    if warning_hits:
        score += 0.08


    # Critical terms
    if critical_hits:
        score += 0.20


    # --------------------------------------------------------
    # IMPORTANT:
    # Critical complaint + severe complaint
    # should clearly appear as HIGH risk.
    # --------------------------------------------------------

    if severity >= 5 and critical_hits:

        score = max(score, 0.88)

    elif severity >= 4 and critical_hits:

        score = max(score, 0.78)

    elif severity >= 5:

        score = max(score, 0.72)


    score = min(max(score, 0.0), 0.99)

    risk_level = classify_risk(score)


    # --------------------------------------------------------
    # Possible complication
    # --------------------------------------------------------

    complications = {

        "Battery failure":
            "Potential overheating, unexpected shutdown, battery damage or thermal safety event.",

        "Software error":
            "Potential loss of device control, incorrect operation or interruption during use.",

        "Mechanical defect":
            "Potential component breakage, device malfunction or physical injury risk.",

        "Labeling issue":
            "Potential incorrect setup or operation caused by unclear instructions.",

        "Connector failure":
            "Potential signal interruption, loss of connectivity or device malfunction.",

        "Sensor inaccuracy":
            "Potential incorrect measurements leading to inappropriate device response or decision-making.",

        "User error":
            "Potential incorrect operation or use outside intended operating conditions.",

        "Packaging defect":
            "Potential contamination, loss of sterility or compromised device integrity."
    }


    complication = complications.get(
        category,
        "Potential device malfunction requiring further investigation."
    )


    # More specific complication for critical text
    if any(
        word in text
        for word in [
            "heating",
            "overheating",
            "burn",
            "smoke",
            "fire"
        ]
    ):

        complication = (
            "Potential thermal event: overheating may lead to "
            "device shutdown, component damage, burns or fire-related safety concerns."
        )


    if "shock" in text:

        complication = (
            "Potential electrical safety event: electrical leakage or "
            "fault may expose the user or patient to electric shock."
        )


    if any(
        word in text
        for word in [
            "incorrect reading",
            "wrong reading",
            "inaccurate"
        ]
    ):

        complication = (
            "Potential measurement error: inaccurate device output "
            "could affect clinical interpretation or subsequent action."
        )


    # --------------------------------------------------------
    # Recommended action
    # --------------------------------------------------------

    if risk_level == "HIGH":

        action = (
            "Immediate review recommended. Verify the device condition, "
            "investigate the failure mode and assess whether corrective "
            "or preventive action is required."
        )

    elif risk_level == "MODERATE":

        action = (
            "Further monitoring recommended. Review similar complaints "
            "and investigate whether the pattern is increasing."
        )

    else:

        action = (
            "Continue routine monitoring and record the complaint "
            "for future trend analysis."
        )


    return {
        "score": score,
        "level": risk_level,
        "complication": complication,
        "action": action,
        "critical_hits": critical_hits,
        "warning_hits": warning_hits,
        "baseline": baseline
    }


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">MedRisk AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Medical Device Risk Intelligence Platform</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="description">
        Analyse medical device complaints, identify known failure modes
        and detect potential emerging safety risks using AI-assisted analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🔄 How MedRisk AI Works"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            ### 1️⃣ Collect

            Enter medical device complaint information,
            category, severity and description.
            """
        )

    with c2:

        st.markdown(
            """
            ### 2️⃣ Analyse

            Analyse complaint severity, failure category,
            historical trends and known risks.
            """
        )

    with c3:

        st.markdown(
            """
            ### 3️⃣ Predict

            Generate an AI-assisted risk score and
            identify potential safety concerns.
            """
        )

    st.markdown(
        '<div class="section-heading">Current System Overview</div>',
        unsafe_allow_html=True
    )

    device_count = len(devices)

    complaint_count = (
        len(complaints)
        if not complaints.empty
        else 0
    )

    fmea_count = 0

    try:

        fmea = pd.read_csv(
            "data/fmea_risk_items.csv"
        )

        fmea_count = len(fmea)

    except Exception:
        fmea_count = 0


    a, b, c = st.columns(3)

    with a:

        st.markdown(
            f"""
            <div class="info-card">
            <div class="card-title">Devices Monitored</div>
            <div class="card-value">{device_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            f"""
            <div class="info-card">
            <div class="card-title">Complaint Records</div>
            <div class="card-value">{complaint_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c:

        st.markdown(
            f"""
            <div class="info-card">
            <div class="card-title">Known FMEA Risks</div>
            <div class="card-value">{fmea_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "💡 Go to 'Report Complaint' to test the live AI-assisted risk assessment."
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
        """
        <div class="description">
        Enter information about a medical device complaint to generate
        an AI-assisted risk assessment.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Device Information</div>',
        unsafe_allow_html=True
    )

    device = st.selectbox(
        "Medical Device",
        devices
    )

    complaint_date = st.date_input(
        "Complaint Date",
        value=date.today()
    )


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
        f"Selected severity: {severity_names[severity]} ({severity}/5)"
    )


    description = st.text_area(
        "Describe the complaint",
        placeholder=(
            "Example: Battery is heating during operation and "
            "the device shuts down unexpectedly."
        ),
        height=130
    )


    assess = st.button(
        "🔎 Assess Device Risk",
        type="primary",
        use_container_width=True
    )


    if assess:

        if not description.strip():

            st.warning(
                "Please enter a complaint description."
            )

        else:

            result = analyse_complaint(
                device,
                category,
                severity,
                description
            )

            st.markdown("---")

            st.markdown(
                "## 🤖 AI-Assisted Risk Assessment"
            )


            level = result["level"]
            score = result["score"]


            # ------------------------------------------------
            # Risk message
            # ------------------------------------------------

            css_class = get_risk_class(level)


            if level == "HIGH":

                message = (
                    f"🔴 HIGH RISK — Immediate review recommended."
                )

            elif level == "MODERATE":

                message = (
                    f"🟠 MODERATE RISK — Further monitoring is recommended."
                )

            else:

                message = (
                    f"🟢 LOW RISK — Continue routine monitoring."
                )


            st.markdown(
                f"""
                <div class="{css_class}">
                {message}
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown("")


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            m1, m2, m3 = st.columns(3)

            with m1:

                st.metric(
                    "Risk Score",
                    f"{score:.2f}"
                )

            with m2:

                st.metric(
                    "Risk Probability",
                    f"{score * 100:.1f}%"
                )

            with m3:

                st.metric(
                    "Severity",
                    f"{severity}/5"
                )


            # ------------------------------------------------
            # Possible complication
            # ------------------------------------------------

            st.markdown(
                '<div class="section-heading">⚠️ Possible Complication</div>',
                unsafe_allow_html=True
            )

            st.warning(
                result["complication"]
            )


            # ------------------------------------------------
            # Why the system flagged it
            # ------------------------------------------------

            st.markdown(
                '<div class="section-heading">🔍 Why was this flagged?</div>',
                unsafe_allow_html=True
            )

            reasons = []

            if severity >= 4:

                reasons.append(
                    f"High complaint severity ({severity}/5)"
                )

            if result["critical_hits"]:

                reasons.append(
                    "Critical safety indicators detected in complaint text: "
                    + ", ".join(result["critical_hits"])
                )

            if result["warning_hits"]:

                reasons.append(
                    "Additional warning indicators detected: "
                    + ", ".join(result["warning_hits"])
                )

            if category in [
                "Battery failure",
                "Mechanical defect",
                "Connector failure",
                "Sensor inaccuracy"
            ]:

                reasons.append(
                    f"Failure category identified as: {category}"
                )

            if not reasons:

                reasons.append(
                    "Risk assessment is based on complaint severity, "
                    "failure category and historical device risk."
                )

            for reason in reasons:

                st.write(
                    f"• {reason}"
                )


            # ------------------------------------------------
            # Recommended action
            # ------------------------------------------------

            st.markdown(
                '<div class="section-heading">🛠 Recommended Action</div>',
                unsafe_allow_html=True
            )

            st.info(
                result["action"]
            )


            # ------------------------------------------------
            # Save live assessment
            # ------------------------------------------------

            assessment_row = pd.DataFrame(
                [{
                    "date": complaint_date,
                    "device": device,
                    "category": category,
                    "severity": severity,
                    "description": description,
                    "risk_score": round(score, 3),
                    "risk_level": level,
                    "possible_complication": result["complication"]
                }]
            )


            output_file = (
                "outputs/live_assessments.csv"
            )


            try:

                if os.path.exists(output_file):

                    old = pd.read_csv(
                        output_file
                    )

                    combined = pd.concat(
                        [old, assessment_row],
                        ignore_index=True
                    )

                else:

                    combined = assessment_row


                combined.to_csv(
                    output_file,
                    index=False
                )

            except Exception:

                pass


            st.success(
                "Risk assessment completed successfully."
            )


# ============================================================
# RISK DASHBOARD
# ============================================================

elif page == "📊 Risk Dashboard":

    st.markdown(
        '<div class="main-title">Device Risk Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="description">
        Monitor historical complaint trends and predicted device risk.
        </div>
        """,
        unsafe_allow_html=True
    )


    selected_device = st.selectbox(
        "Select a medical device",
        devices
    )


    if not risk_scores.empty:

        sub = risk_scores[
            risk_scores["product_id"] == selected_device
        ].sort_values("date")


        if not sub.empty:

            latest = sub.iloc[-1]

            score = float(
                latest["risk_score"]
            )

            complaints_30 = int(
                sub["complaint_count"].tail(30).sum()
            )

            high_sev_30 = int(
                sub["high_severity_count"].tail(30).sum()
            )


            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Current Risk Score",
                    f"{score:.2f}"
                )

            with c2:

                st.metric(
                    "Complaints",
                    complaints_30
                )

            with c3:

                st.metric(
                    "High Severity",
                    high_sev_30
                )


            level = classify_risk(score)


            if level == "HIGH":

                st.error(
                    f"🔴 Current predicted risk: HIGH ({score:.2f})"
                )

            elif level == "MODERATE":

                st.warning(
                    f"🟠 Current predicted risk: MODERATE ({score:.2f})"
                )

            else:

                st.success(
                    f"🟢 Current predicted risk: LOW ({score:.2f})"
                )


            st.markdown(
                '<div class="section-heading">📈 Risk Score vs Complaint Volume</div>',
                unsafe_allow_html=True
            )


            chart_data = sub[
                [
                    "date",
                    "risk_score",
                    "complaint_count"
                ]
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


        else:

            st.info(
                "No risk data available for this device."
            )

    else:

        st.warning(
            "Risk prediction data is not available yet."
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
        """
        <div class="description">
        Detect groups of similar complaints that do not match
        documented FMEA risks well.
        </div>
        """,
        unsafe_allow_html=True
    )


    if not emerging.empty:

        st.success(
            f"{len(emerging)} potential emerging-risk pattern(s) detected."
        )


        for _, row in emerging.iterrows():

            st.warning(
                f"""
                **Device:** {row.get('product_id', 'Unknown')}

                **Cluster:** {row.get('cluster_id', 'Unknown')}

                **Similar complaints:** {row.get('num_complaints', 0)}

                **Example:** {row.get('sample_text', 'N/A')}
                """
            )

    else:

        st.info(
            "No emerging risk clusters are currently available."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">About MedRisk AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### 🎯 Project Objective

        MedRisk AI is a prototype platform for monitoring
        medical device complaints and identifying potential
        safety risks.

        ### 🔄 System Workflow

        **Complaint → Feature Analysis → Risk Prediction → Risk Level → Possible Complication**

        ### 🧠 AI / ML Components

        - Complaint trend analysis
        - Historical risk prediction
        - Complaint severity analysis
        - Text-based warning detection
        - Complaint-to-FMEA matching
        - Emerging risk clustering

        ### 📊 Data

        The current prototype uses synthetic complaint data
        for demonstration and testing.

        ### ⚠️ Important Limitation

        This system is a student research prototype and has
        not been clinically or regulatorily validated.
        It must not be used for real medical or regulatory
        decision-making.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "MedRisk AI | Medical Device Risk Intelligence Prototype | "
    "Synthetic data demonstration"
)

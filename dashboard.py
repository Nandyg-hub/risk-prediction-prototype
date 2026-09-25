import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MedRisk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    padding-top: 3.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1250px;
}

/* Main headings */

h1 {
    color: #12355b !important;
    font-weight: 700 !important;
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

/* Inputs */

input,
textarea {
    background-color: white !important;
    color: #1e293b !important;
}

[data-baseweb="select"] {
    background-color: white !important;
}

/* Navigation */

.nav-title {
    font-size: 30px;
    font-weight: 800;
    color: #12355b;
    margin-bottom: 2px;
}

.nav-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 18px;
}

/* Cards */

.info-card {
    background: white;
    border: 1px solid #dbe3ec;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 15px;
}

.result-card {
    background: white;
    border: 1px solid #dbe3ec;
    border-radius: 14px;
    padding: 24px;
    margin-top: 15px;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: white !important;
    border: 1px solid #dbe3ec;
    padding: 18px;
    border-radius: 14px;
}

/* Metric text */

[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

[data-testid="stMetricValue"] {
    color: #12355b !important;
}

/* Buttons */

.stButton > button {
    border-radius: 9px;
    font-weight: 600;
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
# DIRECTORIES
# =========================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================================================
# AUTOMATIC FIRST-TIME SETUP
# =========================================================

REQUIRED_FILES = [
    "data/complaints.csv",
    "data/fmea_risk_items.csv",
    "outputs/risk_scores_full.csv",
    "outputs/complaint_matches.csv"
]

if not all(os.path.exists(file) for file in REQUIRED_FILES):

    with st.spinner("Setting up MedRisk AI..."):

        scripts = [
            "generate_synthetic_data.py",
            "generate_fmea.py",
            "risk_prediction.py",
            "complaint_matching.py"
        ]

        for script in scripts:

            if not os.path.exists(script):
                continue

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

    complaints = pd.read_csv(
        "data/complaints.csv",
        parse_dates=["date"]
    )

    fmea = pd.read_csv(
        "data/fmea_risk_items.csv"
    )

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

    return complaints, fmea, risk_scores, matches, emerging


complaints, fmea, risk_scores, matches, emerging = load_data()


products = sorted(
    complaints["product_id"].unique()
)


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURE_COLS = [
    "complaint_count",
    "avg_severity",
    "max_severity",
    "high_severity_count",
    "roll7_count",
    "roll30_count",
    "roll30_high_sev",
    "roll30_avg_severity",
    "trend_7_vs_30"
]


# =========================================================
# TRAIN RANDOM FOREST
# Same methodology as risk_prediction.py
# =========================================================

@st.cache_resource
def train_model(complaint_data):

    df = complaint_data.copy()

    all_dates = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D"
    )

    product_list = df["product_id"].unique()

    daily = (
        df.groupby(["product_id", "date"])
        .agg(
            complaint_count=("complaint_id", "count"),
            avg_severity=("severity", "mean"),
            max_severity=("severity", "max"),
            high_severity_count=(
                "severity",
                lambda s: (s >= 4).sum()
            )
        )
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [product_list, all_dates],
        names=["product_id", "date"]
    )

    daily_full = (
        daily
        .set_index(["product_id", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    daily_full["avg_severity"] = (
        daily_full["avg_severity"]
        .fillna(0)
    )

    daily_full["max_severity"] = (
        daily_full["max_severity"]
        .fillna(0)
    )

    daily_full = daily_full.sort_values(
        ["product_id", "date"]
    )

    feature_frames = []

    for pid, group in daily_full.groupby("product_id"):

        group = group.sort_values(
            "date"
        ).copy()

        group["roll7_count"] = (
            group["complaint_count"]
            .rolling(7, min_periods=1)
            .sum()
        )

        group["roll30_count"] = (
            group["complaint_count"]
            .rolling(30, min_periods=1)
            .sum()
        )

        group["roll30_high_sev"] = (
            group["high_severity_count"]
            .rolling(30, min_periods=1)
            .sum()
        )

        group["roll30_avg_severity"] = (
            group["avg_severity"]
            .rolling(30, min_periods=1)
            .mean()
        )

        group["trend_7_vs_30"] = (
            group["roll7_count"] * (30 / 7)
        ) - group["roll30_count"]

        feature_frames.append(group)

    feat = pd.concat(
        feature_frames,
        ignore_index=True
    )

    # Future event label

    event_frames = []

    for pid, group in feat.groupby("product_id"):

        group = group.sort_values(
            "date"
        ).copy()

        group["high_sev_7d"] = (
            group["high_severity_count"]
            .rolling(10, min_periods=1)
            .sum()
        )

        group["is_event_day"] = (
            group["high_sev_7d"] >= 2
        ).astype(int)

        reversed_event = (
            group["is_event_day"][::-1]
        )

        future_window = (
            reversed_event
            .rolling(30, min_periods=1)
            .max()[::-1]
        )

        group["label_future_risk"] = (
            future_window
            .shift(-1)
            .fillna(0)
            .astype(int)
        )

        event_frames.append(group)

    labeled = pd.concat(
        event_frames,
        ignore_index=True
    )

    split_date = labeled["date"].quantile(
        0.65
    )

    train_df = labeled[
        labeled["date"] <= split_date
    ]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label_future_risk"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


model = train_model(
    complaints
)


# =========================================================
# LIVE FEATURE CALCULATION
# =========================================================

def calculate_live_features(
    original_data,
    product_id,
    severity,
    prediction_date
):

    data = original_data.copy()

    prediction_date = pd.Timestamp(
        prediction_date
    )

    product_data = data[
        data["product_id"] == product_id
    ].copy()

    # Add the live complaint temporarily

    new_complaint = pd.DataFrame({

        "complaint_id": [
            "LIVE-COMPLAINT"
        ],

        "date": [
            prediction_date
        ],

        "product_id": [
            product_id
        ],

        "category": [
            "user_reported"
        ],

        "severity": [
            int(severity)
        ],

        "description": [
            "Live user complaint"
        ]
    })

    product_data = pd.concat(
        [
            product_data,
            new_complaint
        ],
        ignore_index=True
    )

    all_dates = pd.date_range(
        product_data["date"].min(),
        product_data["date"].max(),
        freq="D"
    )

    daily = (
        product_data
        .groupby("date")
        .agg(
            complaint_count=(
                "complaint_id",
                "count"
            ),

            avg_severity=(
                "severity",
                "mean"
            ),

            max_severity=(
                "severity",
                "max"
            ),

            high_severity_count=(
                "severity",
                lambda s: (s >= 4).sum()
            )
        )
        .reindex(
            all_dates,
            fill_value=0
        )
        .reset_index()
    )

    daily = daily.rename(
        columns={
            "index": "date"
        }
    )

    daily["avg_severity"] = (
        daily["avg_severity"]
        .fillna(0)
    )

    daily["max_severity"] = (
        daily["max_severity"]
        .fillna(0)
    )

    daily["roll7_count"] = (
        daily["complaint_count"]
        .rolling(7, min_periods=1)
        .sum()
    )

    daily["roll30_count"] = (
        daily["complaint_count"]
        .rolling(30, min_periods=1)
        .sum()
    )

    daily["roll30_high_sev"] = (
        daily["high_severity_count"]
        .rolling(30, min_periods=1)
        .sum()
    )

    daily["roll30_avg_severity"] = (
        daily["avg_severity"]
        .rolling(30, min_periods=1)
        .mean()
    )

    daily["trend_7_vs_30"] = (
        daily["roll7_count"] * (30 / 7)
    ) - daily["roll30_count"]

    live_row = daily[
        daily["date"] == prediction_date
    ].iloc[-1]

    return pd.DataFrame(
        [
            live_row[FEATURE_COLS].values
        ],
        columns=FEATURE_COLS
    )


# =========================================================
# LIVE COMPLAINT → FMEA MATCHING
# =========================================================

def match_live_complaint(
    complaint_text,
    product_id
):

    product_fmea = fmea[
        fmea["product_id"] == product_id
    ].copy()

    if product_fmea.empty:

        return {
            "matched": False,
            "risk_id": None,
            "category": "Unknown",
            "description": None,
            "confidence": 0.0
        }

    texts = (
        product_fmea["description"]
        .fillna("")
        .tolist()
    )

    corpus = [
        complaint_text
    ] + texts

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            corpus
        )

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        )[0]

        best_index = int(
            np.argmax(similarity)
        )

        best_score = float(
            similarity[best_index]
        )

    except Exception:

        best_index = 0
        best_score = 0.0

    best_row = product_fmea.iloc[
        best_index
    ]

    return {
        "matched": best_score >= 0.15,
        "risk_id": best_row["risk_id"],
        "category": best_row.get(
            "category",
            "Known risk"
        ),
        "description": best_row["description"],
        "confidence": best_score
    }


# =========================================================
# TEXT RISK SIGNALS
# =========================================================

def analyse_complaint_text(text):

    text = text.lower()

    critical_terms = [
        "patient",
        "critical",
        "life threatening",
        "life-threatening",
        "injury",
        "death",
        "shock",
        "overheating",
        "fire",
        "smoke",
        "unexpected shutdown",
        "repeated shutdown",
        "stopped working",
        "loss of function",
        "failure during operation",
        "failed during operation",
        "malfunction"
    ]

    escalation_terms = [
        "repeated",
        "multiple",
        "frequent",
        "increasing",
        "unexpected",
        "intermittent",
        "again",
        "several",
        "recurring",
        "recently"
    ]

    critical_hits = [
        word
        for word in critical_terms
        if word in text
    ]

    escalation_hits = [
        word
        for word in escalation_terms
        if word in text
    ]

    return {
        "critical_hits": critical_hits,
        "escalation_hits": escalation_hits
    }


# =========================================================
# FINAL LIVE RISK CALCULATION
# =========================================================

def calculate_final_risk(
    ml_score,
    severity,
    complaint_text,
    match_confidence,
    live_features
):

    text_signals = analyse_complaint_text(
        complaint_text
    )

    critical_hits = text_signals[
        "critical_hits"
    ]

    escalation_hits = text_signals[
        "escalation_hits"
    ]

    severity_score = (
        (severity - 1) / 4
    )

    keyword_score = min(
        1.0,
        (
            len(critical_hits) * 0.35
            + len(escalation_hits) * 0.12
        )
    )

    recent_activity = min(
        1.0,
        float(
            live_features[
                "roll30_count"
            ].iloc[0]
        ) / 10.0
    )

    trend_value = float(
        live_features[
            "trend_7_vs_30"
        ].iloc[0]
    )

    trend_score = min(
        1.0,
        max(
            0.0,
            0.5 + trend_value / 10
        )
    )

    # -----------------------------------------------------
    # Prototype risk fusion
    #
    # Combines:
    # 1. Historical Random Forest prediction
    # 2. Complaint severity
    # 3. Text-based safety signals
    # 4. Recent complaint activity
    # 5. Trend
    #
    # This is an AI-assisted prototype index,
    # NOT a clinically validated risk score.
    # -----------------------------------------------------

    final_score = (
        0.40 * ml_score
        + 0.25 * severity_score
        + 0.20 * keyword_score
        + 0.10 * recent_activity
        + 0.05 * trend_score
    )

    # Strong demo safety condition:
    # a critical severity complaint with clear
    # failure/safety language must not be hidden
    # by a low historical ML score.

    if (
        severity >= 5
        and len(critical_hits) >= 1
    ):

        final_score = max(
            final_score,
            0.85
        )

    elif (
        severity >= 4
        and len(critical_hits) >= 2
    ):

        final_score = max(
            final_score,
            0.75
        )

    final_score = min(
        0.99,
        max(
            0.01,
            final_score
        )
    )

    if final_score >= 0.70:

        level = "HIGH"

    elif final_score >= 0.40:

        level = "MODERATE"

    else:

        level = "LOW"

    return (
        float(final_score),
        level,
        critical_hits,
        escalation_hits
    )


# =========================================================
# NAVIGATION
# =========================================================

st.markdown(
    '<div class="nav-title">🏥 MedRisk AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="nav-subtitle">'
    'Medical Device Risk Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)

page = st.radio(
    "Navigation",
    [
        "🏠 Home",
        "📝 Report Complaint",
        "📊 Risk Dashboard",
        "⚠️ Emerging Risks",
        "ℹ️ About"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title(
        "Medical Device Risk Intelligence Platform"
    )

    st.write(
        "Analyse medical device complaints, identify "
        "known failure modes and detect potential "
        "emerging safety risks using AI-assisted analysis."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### 1️⃣ Collect"
        )

        st.write(
            "Enter device complaints, dates "
            "and severity."
        )

    with col2:

        st.markdown(
            "### 2️⃣ Analyse"
        )

        st.write(
            "Compare complaints with known FMEA "
            "risks and historical trends."
        )

    with col3:

        st.markdown(
            "### 3️⃣ Predict"
        )

        st.write(
            "Use machine-learning and complaint "
            "signals to estimate risk."
        )

    st.divider()

    st.subheader(
        "Current System Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Devices Monitored",
            len(products)
        )

    with col2:

        st.metric(
            "Complaint Records",
            len(complaints)
        )

    with col3:

        st.metric(
            "Known FMEA Risks",
            len(fmea)
        )

    st.info(
        "💡 Go to 'Report Complaint' to test "
        "the live AI-assisted risk assessment."
    )


# =========================================================
# REPORT COMPLAINT
# =========================================================

elif page == "📝 Report Complaint":

    st.title(
        "Report a Device Complaint"
    )

    st.write(
        "Enter information about a medical device "
        "complaint to generate an AI-assisted risk assessment."
    )

    st.divider()

    st.subheader(
        "Device Information"
    )

    selected_device = st.selectbox(
        "Medical Device",
        products
    )

    complaint_date = st.date_input(
        "Complaint Date",
        value=pd.Timestamp.today().date()
    )

    st.subheader(
        "Complaint Details"
    )

    complaint_category = st.selectbox(
        "Failure / Complaint Category",
        [
            "Battery failure",
            "Software error",
            "Mechanical defect",
            "Connector failure",
            "Sensor inaccuracy",
            "Packaging defect",
            "Labelling issue",
            "User error",
            "Other"
        ]
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
        5: "Critical"
    }

    st.caption(
        f"Selected severity: "
        f"**{severity_names[severity]} ({severity}/5)**"
    )

    complaint_text = st.text_area(
        "Describe the complaint",
        placeholder=(
            "Example: The device repeatedly shut down "
            "during active operation. Battery drained "
            "unexpectedly and the device stopped functioning."
        ),
        height=160
    )

    st.write("")

    analyse = st.button(
        "🔍 Assess Device Risk",
        type="primary",
        use_container_width=True
    )

    if analyse:

        if not complaint_text.strip():

            st.warning(
                "Please enter a complaint description first."
            )

            st.stop()

        # -------------------------------------------------
        # Dataset currently ends in 2025.
        # For a future demo date, use the next available
        # historical day for the live simulation.
        # -------------------------------------------------

        latest_data_date = complaints["date"].max()

        entered_date = pd.Timestamp(
            complaint_date
        )

        if entered_date > latest_data_date:

            prediction_date = (
                latest_data_date
                + pd.Timedelta(days=1)
            )

            simulated = True

        else:

            prediction_date = entered_date

            simulated = False

        # -------------------------------------------------
        # LIVE MODEL FEATURES
        # -------------------------------------------------

        live_features = calculate_live_features(
            complaints,
            selected_device,
            severity,
            prediction_date
        )

        ml_probability = model.predict_proba(
            live_features[FEATURE_COLS]
        )[0, 1]

        ml_score = float(
            ml_probability
        )

        # -------------------------------------------------
        # LIVE FMEA MATCHING
        # -------------------------------------------------

        fmea_result = match_live_complaint(
            complaint_text,
            selected_device
        )

        # -------------------------------------------------
        # FINAL RISK
        # -------------------------------------------------

        (
            risk_score,
            risk_level,
            critical_hits,
            escalation_hits
        ) = calculate_final_risk(
            ml_score,
            severity,
            complaint_text,
            fmea_result["confidence"],
            live_features
        )

        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        st.session_state[
            "live_prediction"
        ] = {

            "device": selected_device,

            "category": complaint_category,

            "complaint": complaint_text,

            "severity": severity,

            "risk_score": risk_score,

            "risk_level": risk_level,

            "ml_score": ml_score,

            "fmea_result": fmea_result,

            "critical_hits": critical_hits,

            "escalation_hits": escalation_hits
        }

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "🤖 AI-Assisted Risk Assessment"
        )

        if risk_level == "HIGH":

            st.error(
                "🔴 HIGH RISK — Potential emerging "
                "device safety concern detected."
            )

        elif risk_level == "MODERATE":

            st.warning(
                "🟠 MODERATE RISK — Further monitoring "
                "is recommended."
            )

        else:

            st.success(
                "🟢 LOW RISK — No strong escalation "
                "signal detected in this prototype."
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Risk Score",
                f"{risk_score:.2f}"
            )

        with col2:

            st.metric(
                "Risk Probability",
                f"{risk_score * 100:.1f}%"
            )

        with col3:

            st.metric(
                "Severity",
                f"{severity}/5"
            )

        # -------------------------------------------------
        # FAILURE MODE
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "🔎 Detected Failure Mode"
        )

        if fmea_result["matched"]:

            st.write(
                f"**Known Risk:** "
                f"{fmea_result['category']}"
            )

            st.write(
                f"**FMEA Risk ID:** "
                f"{fmea_result['risk_id']}"
            )

            st.write(
                f"**Match Confidence:** "
                f"{fmea_result['confidence']:.2f}"
            )

            with st.expander(
                "View matched FMEA description"
            ):

                st.write(
                    fmea_result["description"]
                )

        else:

            st.warning(
                "No strong match was found in the "
                "documented FMEA risks."
            )

            st.write(
                "This complaint can be reviewed as a "
                "possible undocumented or emerging failure mode."
            )

        # -------------------------------------------------
        # WHY WAS IT FLAGGED?
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "📌 Why was this complaint flagged?"
        )

        indicators = []

        if severity >= 4:

            indicators.append(
                "🔴 High-severity complaint detected"
            )

        if critical_hits:

            indicators.append(
                "⚠️ Safety/failure indicators detected "
                "in the complaint description"
            )

        if escalation_hits:

            indicators.append(
                "📈 Escalation language detected "
                "(repeated/increasing/frequent pattern)"
            )

        roll30 = float(
            live_features[
                "roll30_count"
            ].iloc[0]
        )

        if roll30 > 3:

            indicators.append(
                "📊 Multiple complaints present "
                "in the recent historical window"
            )

        trend = float(
            live_features[
                "trend_7_vs_30"
            ].iloc[0]
        )

        if trend > 0:

            indicators.append(
                "📈 Recent complaint activity "
                "shows an accelerating trend"
            )

        if not indicators:

            indicators.append(
                "🟢 No major escalation indicator "
                "was detected."
            )

        for item in indicators:

            st.write(item)

        # -------------------------------------------------
        # RECOMMENDED ACTION
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "🛠️ Suggested Review Action"
        )

        if risk_level == "HIGH":

            st.error(
                "Review the complaint immediately, "
                "compare with recent device complaints, "
                "verify the failure mode and consider "
                "technical investigation / CAPA review."
            )

        elif risk_level == "MODERATE":

            st.warning(
                "Continue monitoring complaint frequency "
                "and severity and review whether the "
                "pattern is increasing."
            )

        else:

            st.info(
                "Continue routine complaint monitoring "
                "and investigate if similar complaints "
                "begin to accumulate."
            )

        # -------------------------------------------------
        # LIVE MODEL DETAILS
        # -------------------------------------------------

        with st.expander(
            "View technical prediction details"
        ):

            st.write(
                f"Historical ML score: "
                f"**{ml_score:.3f}**"
            )

            st.write(
                f"FMEA similarity: "
                f"**{fmea_result['confidence']:.3f}**"
            )

            st.write(
                f"Recent 30-day complaint count: "
                f"**{roll30:.0f}**"
            )

            st.write(
                f"Trend indicator: "
                f"**{trend:.2f}**"
            )

            st.caption(
                "The displayed risk score is a prototype "
                "risk index combining the trained Random Forest "
                "with explicit complaint severity and text signals. "
                "It is not a clinically validated probability."
            )

        if simulated:

            st.info(
                "ℹ️ Demo note: the selected date is after "
                "the synthetic dataset period, so the live "
                "assessment uses the next available historical "
                "day as a simulation point."
            )


# =========================================================
# RISK DASHBOARD
# =========================================================

elif page == "📊 Risk Dashboard":

    st.title(
        "Device Risk Dashboard"
    )

    st.write(
        "Monitor historical complaint trends and "
        "model-predicted device risk."
    )

    selected_device = st.selectbox(
        "Select a medical device",
        products,
        key="dashboard_device"
    )

    sub = (
        risk_scores[
            risk_scores["product_id"]
            == selected_device
        ]
        .sort_values("date")
    )

    latest = sub.iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Risk Score",
            f"{latest['risk_score']:.2f}"
        )

    with col2:

        st.metric(
            "Complaint Activity",
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

    risk_value = float(
        latest["risk_score"]
    )

    if risk_value >= 0.70:

        st.error(
            f"🔴 Current predicted risk: HIGH ({risk_value:.2f})"
        )

    elif risk_value >= 0.40:

        st.warning(
            f"🟠 Current predicted risk: MODERATE ({risk_value:.2f})"
        )

    else:

        st.success(
            f"🟢 Current predicted risk: LOW ({risk_value:.2f})"
        )

    st.divider()

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
        label="Predicted risk score"
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

    st.subheader(
        "Recent Risk Records"
    )

    display_df = (
        sub[
            [
                "date",
                "risk_score",
                "complaint_count",
                "high_severity_count"
            ]
        ]
        .tail(15)
        .sort_values(
            "date",
            ascending=False
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# EMERGING RISKS
# =========================================================

elif page == "⚠️ Emerging Risks":

    st.title(
        "Emerging / Unknown Risks"
    )

    st.write(
        "Complaints that do not strongly match documented "
        "FMEA risks but show similarity to other complaints "
        "can indicate a possible undocumented failure mode."
    )

    if emerging.empty:

        st.info(
            "No emerging risk clusters are currently available."
        )

    else:

        for product_id in products:

            product_clusters = emerging[
                emerging["product_id"]
                == product_id
            ]

            if product_clusters.empty:

                continue

            st.subheader(
                f"Device {product_id}"
            )

            for _, row in product_clusters.iterrows():

                st.warning(
                    f"**{row['num_complaints']} similar "
                    f"unmatched complaints detected**\n\n"
                    f"Cluster: {row['cluster_id']}\n\n"
                    f"Example: {row['sample_text']}"
                )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.title(
        "About MedRisk AI"
    )

    st.subheader(
        "Medical Device Risk Intelligence Platform"
    )

    st.write(
        """
        MedRisk AI is a student research prototype for
        demonstrating AI-assisted post-market surveillance
        of medical device complaints.
        """
    )

    st.divider()

    st.subheader(
        "System Workflow"
    )

    st.write(
        """
        **1. Complaint Collection**

        A user enters the device, complaint description,
        date and severity.

        **2. Known Risk Matching**

        The complaint is compared with documented FMEA
        failure modes using TF-IDF similarity.

        **3. Historical Risk Prediction**

        A Random Forest model analyses complaint frequency,
        severity and historical trends.

        **4. Live Risk Assessment**

        The prototype combines the historical ML score
        with the current complaint's severity and
        safety-related text indicators.

        **5. Emerging Risk Detection**

        Unmatched complaints can be grouped to identify
        possible undocumented failure patterns.
        """
    )

    st.divider()

    st.subheader(
        "Prototype Scope"
    )

    st.info(
        "All complaint data used by this application is "
        "synthetic. The system is a proof-of-concept and "
        "has not been clinically or regulatorily validated."
    )

    st.divider()

    st.subheader(
        "Technology"
    )

    st.write(
        "Python • Streamlit • Pandas • Scikit-learn • "
        "Random Forest • TF-IDF • FMEA Traceability"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'MedRisk AI • Student Research Prototype • '
    'Synthetic Data • Not for Clinical Decision-Making'
    '</div>',
    unsafe_allow_html=True
)

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE SETUP
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
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

h1 {
    color: #12355b !important;
}

h2 {
    color: #12355b !important;
}

h3 {
    color: #12355b !important;
}

p {
    color: #334155;
}

label {
    color: #334155 !important;
}

input,
textarea {
    background-color: white !important;
    color: #1e293b !important;
}

[data-baseweb="select"] {
    background-color: white !important;
}

[data-testid="stMetric"] {
    background-color: white !important;
    border: 1px solid #dbe3ec;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

[data-testid="stMetricValue"] {
    color: #12355b !important;
}

.footer {
    text-align: center;
    color: #7a869a;
    padding-top: 40px;
    font-size: 13px;
}

.nav-title {
    color: #12355b;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 2px;
}

.nav-subtitle {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DIRECTORIES
# =========================================================

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================================================
# CREATE REQUIRED DATA IF MISSING
# =========================================================

REQUIRED_FILES = [
    "data/complaints.csv",
    "data/fmea_risk_items.csv",
    "outputs/risk_scores_full.csv",
    "outputs/complaint_matches.csv",
]

if not all(os.path.exists(file) for file in REQUIRED_FILES):

    with st.spinner("Setting up MedRisk AI..."):

        scripts = [
            "generate_synthetic_data.py",
            "generate_fmea.py",
            "risk_prediction.py",
            "complaint_matching.py",
        ]

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
                st.error(f"Setup failed while running {script}")
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

    return complaints, risk_scores, matches, emerging


complaints, risk_scores, matches, emerging = load_data()

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
# TRAIN RANDOM FOREST MODEL
# =========================================================

@st.cache_resource
def train_model(complaint_data):

    df = complaint_data.copy()

    # -----------------------------------------------------
    # Daily aggregation
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Full calendar
    # -----------------------------------------------------

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
        daily_full["avg_severity"].fillna(0)
    )

    daily_full["max_severity"] = (
        daily_full["max_severity"].fillna(0)
    )

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    daily_full = daily_full.sort_values(
        ["product_id", "date"]
    )

    feature_frames = []

    for pid, grp in daily_full.groupby("product_id"):

        grp = grp.sort_values("date").copy()

        grp["roll7_count"] = (
            grp["complaint_count"]
            .rolling(7, min_periods=1)
            .sum()
        )

        grp["roll30_count"] = (
            grp["complaint_count"]
            .rolling(30, min_periods=1)
            .sum()
        )

        grp["roll30_high_sev"] = (
            grp["high_severity_count"]
            .rolling(30, min_periods=1)
            .sum()
        )

        grp["roll30_avg_severity"] = (
            grp["avg_severity"]
            .rolling(30, min_periods=1)
            .mean()
        )

        grp["trend_7_vs_30"] = (
            grp["roll7_count"] * (30 / 7)
        ) - grp["roll30_count"]

        feature_frames.append(grp)

    feat = pd.concat(
        feature_frames,
        ignore_index=True
    )

    # -----------------------------------------------------
    # Future risk label
    # -----------------------------------------------------

    event_frames = []

    for pid, grp in feat.groupby("product_id"):

        grp = grp.sort_values("date").copy()

        grp["high_sev_7d"] = (
            grp["high_severity_count"]
            .rolling(10, min_periods=1)
            .sum()
        )

        grp["is_event_day"] = (
            grp["high_sev_7d"] >= 2
        ).astype(int)

        reversed_event = (
            grp["is_event_day"][::-1]
        )

        future_window = (
            reversed_event
            .rolling(30, min_periods=1)
            .max()[::-1]
        )

        grp["label_future_risk"] = (
            future_window
            .shift(-1)
            .fillna(0)
            .astype(int)
        )

        event_frames.append(grp)

    labeled = pd.concat(
        event_frames,
        ignore_index=True
    )

    # -----------------------------------------------------
    # Time based train split
    # -----------------------------------------------------

    split_date = labeled["date"].quantile(0.65)

    train_df = labeled[
        labeled["date"] <= split_date
    ]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label_future_risk"]

    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

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


model = train_model(complaints)


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

    # -----------------------------------------------------
    # Selected product
    # -----------------------------------------------------

    product_data = data[
        data["product_id"] == product_id
    ].copy()

    # -----------------------------------------------------
    # Add current complaint temporarily
    # -----------------------------------------------------

    new_complaint = pd.DataFrame({
        "complaint_id": ["LIVE-COMPLAINT"],
        "date": [prediction_date],
        "product_id": [product_id],
        "category": ["user_reported"],
        "severity": [int(severity)],
        "description": ["Live user complaint"]
    })

    product_data = pd.concat(
        [
            product_data,
            new_complaint
        ],
        ignore_index=True
    )

    # -----------------------------------------------------
    # Date range
    # -----------------------------------------------------

    start_date = product_data["date"].min()

    end_date = max(
        product_data["date"].max(),
        prediction_date
    )

    all_dates = pd.date_range(
        start_date,
        end_date,
        freq="D"
    )

    # -----------------------------------------------------
    # Daily aggregation
    # -----------------------------------------------------

    daily = (
        product_data
        .groupby("date")
        .agg(
            complaint_count=("complaint_id", "count"),
            avg_severity=("severity", "mean"),
            max_severity=("severity", "max"),
            high_severity_count=(
                "severity",
                lambda s: (s >= 4).sum()
            )
        )
        .reindex(all_dates, fill_value=0)
        .reset_index()
    )

    daily = daily.rename(
        columns={"index": "date"}
    )

    daily["avg_severity"] = (
        daily["avg_severity"].fillna(0)
    )

    daily["max_severity"] = (
        daily["max_severity"].fillna(0)
    )

    # -----------------------------------------------------
    # Rolling features
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Get live row
    # -----------------------------------------------------

    live_row = daily[
        daily["date"] == prediction_date
    ].iloc[-1]

    feature_row = pd.DataFrame(
        [live_row[FEATURE_COLS].values],
        columns=FEATURE_COLS
    )

    return feature_row


# =========================================================
# TOP NAVIGATION
# =========================================================

st.markdown("""
<div class="nav-title">
🏥 MedRisk AI
</div>

<div class="nav-subtitle">
Medical Device Risk Intelligence Platform
</div>
""", unsafe_allow_html=True)


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

    st.title("MedRisk AI")

    st.subheader(
        "Medical Device Risk Intelligence Platform"
    )

    st.write(
        "Analyse medical device complaints, identify known "
        "failure modes and detect potential emerging safety risks."
    )

    st.write("")

    # -----------------------------------------------------
    # HOW IT WORKS
    # -----------------------------------------------------

    st.subheader("How MedRisk AI Works")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 1️⃣ Collect")

        st.write(
            "Enter device complaints, dates and severity."
        )

    with col2:

        st.markdown("### 2️⃣ Analyse")

        st.write(
            "Compare complaints with known FMEA risks "
            "and analyse historical complaint trends."
        )

    with col3:

        st.markdown("### 3️⃣ Predict")

        st.write(
            "Use machine learning to estimate future "
            "device risk."
        )

    st.divider()

    # -----------------------------------------------------
    # SYSTEM OVERVIEW
    # -----------------------------------------------------

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
            len(complaints)
        )

    with col3:

        st.metric(
            "Risk Records",
            len(risk_scores)
        )

    st.info(
        "💡 Use **Report Complaint** above to enter a "
        "new complaint and generate a live AI risk assessment."
    )


# =========================================================
# REPORT COMPLAINT
# =========================================================

elif page == "📝 Report Complaint":

    st.title("Report a Device Complaint")

    st.write(
        "Enter information about a medical device complaint "
        "to generate an AI-assisted risk assessment."
    )

    st.divider()

    # -----------------------------------------------------
    # DEVICE INFORMATION
    # -----------------------------------------------------

    st.subheader("Device Information")

    selected_device = st.selectbox(
        "Medical Device",
        products
    )

    complaint_date = st.date_input(
        "Complaint Date"
    )

    # -----------------------------------------------------
    # COMPLAINT
    # -----------------------------------------------------

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

    # =====================================================
    # PREDICTION
    # =====================================================

    if analyse:

        if not complaint_text.strip():

            st.warning(
                "⚠️ Please enter a complaint description first."
            )

        else:

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
            # Calculate live features
            # -------------------------------------------------

            live_features = calculate_live_features(
                complaints,
                selected_device,
                severity,
                prediction_date
            )

            # -------------------------------------------------
            # AI prediction
            # -------------------------------------------------

            risk_probability = model.predict_proba(
                live_features[FEATURE_COLS]
            )[0, 1]

            risk_score = float(
                risk_probability
            )

            # -------------------------------------------------
            # Risk category
            # -------------------------------------------------

            if risk_score >= 0.70:

                risk_level = "HIGH"

            elif risk_score >= 0.40:

                risk_level = "MODERATE"

            else:

                risk_level = "LOW"

            # -------------------------------------------------
            # Save session result
            # -------------------------------------------------

            st.session_state["live_prediction"] = {
                "device": selected_device,
                "complaint": complaint_text,
                "severity": severity,
                "risk_score": risk_score,
                "risk_level": risk_level
            }

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "🤖 AI Risk Assessment"
            )

            if risk_level == "HIGH":

                st.error(
                    f"🔴 Predicted Risk: **{risk_level}**"
                )

            elif risk_level == "MODERATE":

                st.warning(
                    f"🟠 Predicted Risk: **{risk_level}**"
                )

            else:

                st.success(
                    f"🟢 Predicted Risk: **{risk_level}**"
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
            # PREDICTION INDICATORS
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "📌 Prediction Indicators"
            )

            high_severity = float(
                live_features[
                    "high_severity_count"
                ].iloc[0]
            )

            recent_complaints = float(
                live_features[
                    "roll7_count"
                ].iloc[0]
            )

            monthly_complaints = float(
                live_features[
                    "roll30_count"
                ].iloc[0]
            )

            trend = float(
                live_features[
                    "trend_7_vs_30"
                ].iloc[0]
            )

            indicators = []

            if severity >= 4:

                indicators.append(
                    "🔴 High-severity complaint detected"
                )

            if high_severity > 0:

                indicators.append(
                    "⚠️ High-severity activity present"
                )

            if recent_complaints > 1:

                indicators.append(
                    "📈 Recent complaint activity detected"
                )

            if monthly_complaints > 3:

                indicators.append(
                    "📊 Multiple complaints in the recent 30-day window"
                )

            if trend > 0:

                indicators.append(
                    "📈 Complaint trend is accelerating"
                )

            if not indicators:

                indicators.append(
                    "🟢 No major escalation indicator detected "
                    "in the available historical features"
                )

            for indicator in indicators:

                st.write(indicator)

            # -------------------------------------------------
            # PROTOTYPE NOTICE
            # -------------------------------------------------

            if simulated:

                st.info(
                    "ℹ️ Prototype simulation: the historical "
                    f"dataset ends on "
                    f"{latest_data_date.strftime('%d %b %Y')}. "
                    "The new complaint is evaluated as the next "
                    "available observation. The original dataset "
                    "is not modified."
                )

            else:

                st.info(
                    "ℹ️ This prediction is temporary. "
                    "The original complaint dataset is not modified."
                )

            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "Complaint Summary"
            )

            st.write(
                f"**Device:** {selected_device}"
            )

            st.write(
                f"**Complaint:** {complaint_text}"
            )

            st.write(
                f"**Severity:** "
                f"{severity_names[severity]} ({severity}/5)"
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

    risk_score = float(
        latest["risk_score"]
    )

    if risk_score >= 0.70:

        st.error(
            f"🔴 Current predicted risk: "
            f"**HIGH ({risk_score:.2f})**"
        )

    elif risk_score >= 0.40:

        st.warning(
            f"🟠 Current predicted risk: "
            f"**MODERATE ({risk_score:.2f})**"
        )

    else:

        st.success(
            f"🟢 Current predicted risk: "
            f"**LOW ({risk_score:.2f})**"
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

    plt.close(fig)

    # -----------------------------------------------------
    # MATCHING
    # -----------------------------------------------------

    st.divider()

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

            available_columns = [
                col
                for col in [
                    "complaint_id",
                    "complaint_text",
                    "best_match_risk_id",
                    "best_match_text",
                    "confidence"
                ]
                if col in matched_data.columns
            ]

            st.dataframe(
                matched_data[
                    available_columns
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
    - 📝 Interactive complaint assessment
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

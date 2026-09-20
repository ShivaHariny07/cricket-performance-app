"""
Cricket Performance Analysis & Prediction System
--------------------------------------------------
Web application built with Streamlit that wraps the pre-trained
Random Forest models (player-level and team-level) produced by the
modelling half of this team mini-project.

Two ways to get predictions:
  1. Manual entry   -> fill a form for a single player / pick a team
  2. CSV upload     -> upload a file and get predictions for every row

Run with:  streamlit run app.py
"""

import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ================================================================
# PAGE CONFIG (must be the first Streamlit call)
# ================================================================
st.set_page_config(
    page_title="Cricket Performance Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CONSTANTS
# ================================================================
MODEL_DIR = "models"
DATA_DIR = "dataset"

NUMERIC_COLUMNS = [
    "total_runs",
    "strike_rate",
    "total_balls_faced",
    "total_wickets_taken",
    "total_runs_conceded",
    "total_overs_bowled",
    "total_matches_played",
    "matches_played_as_batter",
    "matches_played_as_bowler",
    "matches_won",
    "matches_lost",
    "player_of_match_awards",
    "average",
]

RESULT_COLORS = {"Good": "#16a34a", "Average": "#d97706", "Poor": "#dc2626"}
RESULT_ICON = {"Good": "🟢", "Average": "🟡", "Poor": "🔴"}

PLAYER_TEMPLATE_COLUMNS = [
    "player_name", "role", "team", "total_runs", "strike_rate",
    "total_balls_faced", "average", "player_of_match_awards",
    "total_wickets_taken", "total_runs_conceded", "total_overs_bowled",
    "total_matches_played", "matches_played_as_batter",
    "matches_played_as_bowler", "matches_won", "matches_lost",
]

# ================================================================
# STYLING
# ================================================================
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .main { background: #f5f7fb; }

        /* Hero banner */
        .hero {
            background: linear-gradient(120deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
            padding: 2.6rem 2.4rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 30px rgba(15, 32, 39, 0.25);
        }
        .hero h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 2.3rem;
            margin-bottom: 0.35rem;
        }
        .hero p {
            font-size: 1.02rem;
            opacity: 0.88;
            margin: 0;
        }
        .hero .badge {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.8rem;
            text-transform: uppercase;
        }

        /* Section cards */
        .card {
            background: white;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 4px 18px rgba(20, 20, 43, 0.06);
            border: 1px solid #eef0f5;
            margin-bottom: 1.1rem;
        }
        .section-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.05rem;
            color: #1f2937;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Result banner */
        .result-box {
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            color: white;
            font-family: 'Poppins', sans-serif;
            box-shadow: 0 10px 24px rgba(0,0,0,0.12);
        }
        .result-box h2 { margin: 0 0 0.2rem 0; font-size: 1.6rem; }
        .result-box p { margin: 0; opacity: 0.92; }

        /* Metric-style pill */
        .pill {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: white;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101828 0%, #1d2939 100%);
        }
        div[data-testid="stSidebar"] * { color: #e5e7eb !important; }
        div[data-testid="stSidebar"] .stRadio label { font-weight: 500; }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
            border: none;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(120deg, #2c5364, #0f2027);
        }

        [data-testid="stMetricValue"] { font-family: 'Poppins', sans-serif; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def result_box_html(title, subtitle):
    color = RESULT_COLORS.get(title, "#374151")
    icon = RESULT_ICON.get(title, "ℹ️")
    st.markdown(
        f"""
        <div class="result-box" style="background:{color};">
            <h2>{icon} {title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# DATA / MODEL LOADING (cached)
# ================================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    artifacts = {
        "player_model": joblib.load(f"{MODEL_DIR}/random_forest_performance_model.pkl"),
        "role_encoder": joblib.load(f"{MODEL_DIR}/role_encoder.pkl"),
        "team_encoder": joblib.load(f"{MODEL_DIR}/team_encoder.pkl"),
        "player_features": joblib.load(f"{MODEL_DIR}/feature_columns.pkl"),
        "team_model": joblib.load(f"{MODEL_DIR}/team_random_forest_performance_model.pkl"),
        "team_features": joblib.load(f"{MODEL_DIR}/team_feature_columns.pkl"),
    }
    return artifacts


def clean_numeric(df, columns):
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            if df[column].isnull().sum() > 0:
                median = df[column].median()
                df[column] = df[column].fillna(median if not np.isnan(median) else 0)
    return df


@st.cache_data(show_spinner=False)
def load_default_dataset():
    raw = pd.read_csv(f"{DATA_DIR}/ODI_Cricket_Data_new.csv")
    raw = clean_numeric(raw, NUMERIC_COLUMNS)
    team_df = raw.groupby("team").agg({
        "total_runs": "sum",
        "strike_rate": "mean",
        "total_balls_faced": "sum",
        "total_wickets_taken": "sum",
        "total_runs_conceded": "sum",
        "total_overs_bowled": "sum",
        "total_matches_played": "sum",
        "matches_played_as_batter": "sum",
        "matches_played_as_bowler": "sum",
        "matches_won": "sum",
        "matches_lost": "sum",
        "player_of_match_awards": "sum",
        "average": "mean",
    }).reset_index()
    return raw, team_df


def predict_players(df, artifacts):
    """Vectorised prediction for a dataframe of player rows.
    Returns (results_df, skipped_df) where skipped rows have a role/team
    the encoders have never seen."""
    role_encoder = artifacts["role_encoder"]
    team_encoder = artifacts["team_encoder"]
    features = artifacts["player_features"]
    model = artifacts["player_model"]

    df = clean_numeric(df, NUMERIC_COLUMNS)

    known_role = df["role"].astype(str).isin(role_encoder.classes_)
    known_team = df["team"].astype(str).isin(team_encoder.classes_)
    valid_mask = known_role & known_team

    valid_df = df[valid_mask].copy()
    skipped_df = df[~valid_mask].copy()

    if valid_df.empty:
        return valid_df, skipped_df

    encoded = valid_df.copy()
    encoded["role"] = role_encoder.transform(encoded["role"].astype(str))
    encoded["team"] = team_encoder.transform(encoded["team"].astype(str))

    missing_cols = [c for c in features if c not in encoded.columns]
    for c in missing_cols:
        encoded[c] = 0

    model_input = encoded[features]
    predictions = model.predict(model_input)
    probabilities = model.predict_proba(model_input)

    valid_df["prediction"] = predictions
    for i, cls in enumerate(model.classes_):
        valid_df[f"prob_{cls}"] = (probabilities[:, i] * 100).round(2)

    return valid_df, skipped_df


def predict_teams(team_df, artifacts):
    features = artifacts["team_features"]
    model = artifacts["team_model"]

    df = team_df.copy()
    missing_cols = [c for c in features if c not in df.columns]
    for c in missing_cols:
        df[c] = 0

    model_input = df[features]
    predictions = model.predict(model_input)
    probabilities = model.predict_proba(model_input)

    df["prediction"] = predictions
    for i, cls in enumerate(model.classes_):
        df[f"prob_{cls}"] = (probabilities[:, i] * 100).round(2)

    return df


def to_csv_bytes(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def result_distribution_chart(df, label_col="prediction", title="Predicted performance mix"):
    counts = df[label_col].value_counts().reindex(["Good", "Average", "Poor"]).fillna(0)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[RESULT_COLORS.get(c, "#6b7280") for c in counts.index],
                text=counts.values.astype(int),
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=340,
        margin=dict(t=50, b=10, l=10, r=10),
        plot_bgcolor="white",
        yaxis_title="Count",
    )
    return fig


# ================================================================
# APP START
# ================================================================
inject_css()

try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Model files not found. Make sure the `models/` folder (with the "
        "six .pkl files) sits next to app.py.\n\n" + str(e)
    )
    st.stop()

try:
    default_raw, default_team_df = load_default_dataset()
    dataset_available = True
except FileNotFoundError:
    default_raw, default_team_df = None, None
    dataset_available = False

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🏏 Cricket AI")
    st.caption("Player & Team Performance Predictor")
    st.divider()
    mode = st.radio(
        "Navigate",
        [
            "🧍 Player — Manual Entry",
            "🏳️ Team — Select & Predict",
            "📁 Batch Upload (CSV)",
            "📊 Dataset Explorer",
        ],
    )
    st.divider()
    st.caption("Model: Random Forest Classifier")
    st.caption("Classes: Good · Average · Poor")
    if dataset_available:
        st.caption(f"Reference dataset: {len(default_raw)} players, {len(default_team_df)} teams")

# ---------------- Hero header ----------------
st.markdown(
    """
    <div class="hero">
        <span class="badge">Python for Data Science · Team Mini Project</span>
        <h1>🏏 Cricket Performance Analysis & Prediction System</h1>
        <p>Predict whether a player or team's career statistics translate into
        <b>Good</b>, <b>Average</b>, or <b>Poor</b> ODI performance — one entry
        at a time, or in bulk from a CSV file.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================================================================
# MODE 1 — PLAYER MANUAL ENTRY
# ================================================================
if mode == "🧍 Player — Manual Entry":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Player Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        role = st.selectbox("Player Role", list(artifacts["role_encoder"].classes_))
    with col2:
        team = st.selectbox("Team", sorted(artifacts["team_encoder"].classes_))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏏 Batting Statistics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        total_runs = st.number_input("Total Runs", min_value=0, value=5000)
    with col2:
        strike_rate = st.number_input("Strike Rate", min_value=0.0, value=90.0)
    with col3:
        total_balls_faced = st.number_input("Total Balls Faced", min_value=0, value=6000)
    col1, col2 = st.columns(2)
    with col1:
        average = st.number_input("Batting Average", min_value=0.0, value=45.0)
    with col2:
        player_of_match_awards = st.number_input("Player of Match Awards", min_value=0, value=10)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Bowling Statistics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        total_wickets_taken = st.number_input("Total Wickets Taken", min_value=0, value=10)
    with col2:
        total_runs_conceded = st.number_input("Total Runs Conceded", min_value=0, value=2000)
    with col3:
        total_overs_bowled = st.number_input("Total Overs Bowled", min_value=0.0, value=300.0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Match Statistics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        total_matches_played = st.number_input("Total Matches Played", min_value=0, value=150)
    with col2:
        matches_played_as_batter = st.number_input("Matches Played as Batter", min_value=0, value=145)
    with col3:
        matches_played_as_bowler = st.number_input("Matches Played as Bowler", min_value=0, value=50)
    col1, col2 = st.columns(2)
    with col1:
        matches_won = st.number_input("Matches Won", min_value=0, value=80)
    with col2:
        matches_lost = st.number_input("Matches Lost", min_value=0, value=60)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔮 Predict Player Performance", type="primary"):
        player_input = pd.DataFrame([{
            "role": role, "total_runs": total_runs, "strike_rate": strike_rate,
            "total_balls_faced": total_balls_faced, "total_wickets_taken": total_wickets_taken,
            "total_runs_conceded": total_runs_conceded, "total_overs_bowled": total_overs_bowled,
            "total_matches_played": total_matches_played,
            "matches_played_as_batter": matches_played_as_batter,
            "matches_played_as_bowler": matches_played_as_bowler,
            "matches_won": matches_won, "matches_lost": matches_lost,
            "player_of_match_awards": player_of_match_awards,
            "average": average, "team": team,
        }])

        results, _ = predict_players(player_input, artifacts)
        prediction = results.iloc[0]["prediction"]
        prob_cols = {c: results.iloc[0][f"prob_{c}"] for c in artifacts["player_model"].classes_}

        st.divider()
        result_box_html(
            prediction,
            f"Based on the entered statistics, this player profile is classified as "
            f"<b>{prediction}</b> performance.",
        )
        st.write("")
        c1, c2, c3 = st.columns(3)
        for col, cls in zip([c1, c2, c3], ["Poor", "Average", "Good"]):
            with col:
                val = prob_cols.get(cls, 0)
                st.metric(f"{RESULT_ICON[cls]} {cls}", f"{val:.2f}%")
                st.progress(int(val))

        fig = go.Figure(go.Bar(
            x=list(prob_cols.keys()),
            y=list(prob_cols.values()),
            marker_color=[RESULT_COLORS.get(c, "#6b7280") for c in prob_cols.keys()],
            text=[f"{v:.1f}%" for v in prob_cols.values()],
            textposition="outside",
        ))
        fig.update_layout(title="Prediction confidence", height=320, plot_bgcolor="white",
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# MODE 2 — TEAM SELECT & PREDICT
# ================================================================
elif mode == "🏳️ Team — Select & Predict":

    if not dataset_available:
        st.warning("The reference dataset (dataset/ODI_Cricket_Data_new.csv) was not found, "
                    "so team aggregates can't be computed. Use Batch Upload instead.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏳️ Select a Team</div>', unsafe_allow_html=True)
        selected_team = st.selectbox("Team", sorted(default_team_df["team"].unique()))
        row = default_team_df[default_team_df["team"] == selected_team]
        display_cols = ["total_runs", "strike_rate", "total_wickets_taken",
                         "matches_won", "matches_lost", "average"]
        st.dataframe(row[display_cols].T.rename(columns={row.index[0]: "Value"}),
                     use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔮 Predict Team Performance", type="primary"):
            result = predict_teams(row, artifacts)
            prediction = result.iloc[0]["prediction"]
            prob_cols = {c: result.iloc[0][f"prob_{c}"] for c in artifacts["team_model"].classes_}

            st.divider()
            result_box_html(prediction, f"<b>{selected_team}</b> is classified as "
                                          f"<b>{prediction}</b> based on aggregated historical statistics.")
            st.write("")
            c1, c2, c3 = st.columns(3)
            for col, cls in zip([c1, c2, c3], ["Poor", "Average", "Good"]):
                with col:
                    val = prob_cols.get(cls, 0)
                    st.metric(f"{RESULT_ICON[cls]} {cls}", f"{val:.2f}%")
                    st.progress(int(val))

        st.divider()
        st.markdown('<div class="section-title">🏆 All Teams Leaderboard</div>', unsafe_allow_html=True)
        if st.button("Rank every team"):
            all_results = predict_teams(default_team_df, artifacts)
            all_results = all_results.sort_values("prob_Good", ascending=False)
            fig = px.bar(
                all_results, x="team", y="prob_Good", color="prediction",
                color_discrete_map=RESULT_COLORS,
                labels={"prob_Good": "Prob. of 'Good' (%)", "team": "Team"},
                title="Teams ranked by predicted 'Good' performance probability",
            )
            fig.update_layout(height=460, plot_bgcolor="white", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                all_results[["team", "prediction", "prob_Good", "prob_Average", "prob_Poor"]],
                use_container_width=True,
            )

# ================================================================
# MODE 3 — BATCH CSV UPLOAD
# ================================================================
elif mode == "📁 Batch Upload (CSV)":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📁 Upload player statistics</div>', unsafe_allow_html=True)
    st.write(
        "Upload a CSV with one row per player. Required columns: "
        f"`{'`, `'.join(PLAYER_TEMPLATE_COLUMNS[1:])}` "
        "(the `player_name` column is optional, used only for display)."
    )

    template_df = pd.DataFrame([{
        "player_name": "Sample Player", "role": "Batter", "team": "India",
        "total_runs": 5000, "strike_rate": 90.0, "total_balls_faced": 6000,
        "average": 45.0, "player_of_match_awards": 10, "total_wickets_taken": 10,
        "total_runs_conceded": 2000, "total_overs_bowled": 300,
        "total_matches_played": 150, "matches_played_as_batter": 145,
        "matches_played_as_bowler": 50, "matches_won": 80, "matches_lost": 60,
    }])
    st.download_button(
        "⬇️ Download CSV template",
        data=to_csv_bytes(template_df),
        file_name="player_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            batch_df = None

        if batch_df is not None:
            missing = [c for c in PLAYER_TEMPLATE_COLUMNS[1:] if c not in batch_df.columns]
            if missing:
                st.error(f"The uploaded file is missing required column(s): {', '.join(missing)}")
            else:
                st.success(f"Loaded {len(batch_df)} rows. Preview below:")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if st.button("🔮 Run Batch Prediction", type="primary"):
                    with st.spinner("Scoring every row..."):
                        results, skipped = predict_players(batch_df, artifacts)

                    if not skipped.empty:
                        st.warning(
                            f"{len(skipped)} row(s) skipped — their `role` or `team` value "
                            "wasn't seen during model training."
                        )
                        with st.expander("View skipped rows"):
                            st.dataframe(skipped, use_container_width=True)

                    if results.empty:
                        st.error("No rows could be scored.")
                    else:
                        st.divider()
                        st.markdown(
                            '<div class="section-title">📊 Batch Results</div>',
                            unsafe_allow_html=True,
                        )

                        c1, c2, c3 = st.columns(3)
                        counts = results["prediction"].value_counts()
                        for col, cls in zip([c1, c2, c3], ["Good", "Average", "Poor"]):
                            with col:
                                st.metric(f"{RESULT_ICON[cls]} {cls}", int(counts.get(cls, 0)))

                        st.plotly_chart(result_distribution_chart(results), use_container_width=True)

                        show_cols = [c for c in ["player_name", "role", "team"] if c in results.columns]
                        show_cols += ["prediction", "prob_Good", "prob_Average", "prob_Poor"]
                        st.dataframe(results[show_cols], use_container_width=True)

                        st.download_button(
                            "⬇️ Download full results as CSV",
                            data=to_csv_bytes(results),
                            file_name="player_predictions.csv",
                            mime="text/csv",
                            type="primary",
                        )

    st.divider()
    with st.expander("📁 Or upload raw match-level data to predict TEAM performance"):
        st.write(
            "Upload a file with the same structure as the original ODI dataset "
            "(one row per player, with a `team` column). The app will aggregate "
            "the statistics per team and run the team-level model."
        )
        team_file = st.file_uploader("Choose a raw dataset CSV", type=["csv"], key="team_csv")
        if team_file is not None:
            try:
                raw = pd.read_csv(team_file)
                raw = clean_numeric(raw, NUMERIC_COLUMNS)
                agg_df = raw.groupby("team").agg({
                    "total_runs": "sum", "strike_rate": "mean", "total_balls_faced": "sum",
                    "total_wickets_taken": "sum", "total_runs_conceded": "sum",
                    "total_overs_bowled": "sum", "total_matches_played": "sum",
                    "matches_played_as_batter": "sum", "matches_played_as_bowler": "sum",
                    "matches_won": "sum", "matches_lost": "sum",
                    "player_of_match_awards": "sum", "average": "mean",
                }).reset_index()

                if st.button("🔮 Run Team Batch Prediction", type="primary"):
                    team_results = predict_teams(agg_df, artifacts)
                    st.plotly_chart(result_distribution_chart(team_results, title="Teams by predicted category"),
                                     use_container_width=True)
                    st.dataframe(
                        team_results[["team", "prediction", "prob_Good", "prob_Average", "prob_Poor"]]
                        .sort_values("prob_Good", ascending=False),
                        use_container_width=True,
                    )
                    st.download_button(
                        "⬇️ Download team results as CSV",
                        data=to_csv_bytes(team_results),
                        file_name="team_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Could not process the file: {e}")

# ================================================================
# MODE 4 — DATASET EXPLORER
# ================================================================
else:
    if not dataset_available:
        st.warning("Reference dataset not found next to app.py.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Players", len(default_raw))
        c2.metric("Teams", default_raw["team"].nunique())
        c3.metric("Avg. Career Runs", f"{default_raw['total_runs'].mean():,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏏 Top Run Scorers</div>', unsafe_allow_html=True)
        top_runs = default_raw.nlargest(15, "total_runs")[["player_name", "team", "total_runs", "average"]]
        fig = px.bar(top_runs.sort_values("total_runs"), x="total_runs", y="player_name",
                     color="team", orientation="h", title="Top 15 run scorers in the dataset")
        fig.update_layout(height=500, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Runs vs Strike Rate</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            default_raw, x="strike_rate", y="total_runs", color="team",
            hover_data=["player_name"], title="Career runs vs strike rate, colored by team",
        )
        fig2.update_layout(height=480, plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🌍 Team Comparison</div>', unsafe_allow_html=True)
        metric_choice = st.selectbox(
            "Compare teams by",
            ["total_runs", "strike_rate", "total_wickets_taken", "matches_won", "average"],
        )
        fig3 = px.bar(
            default_team_df.sort_values(metric_choice, ascending=False),
            x="team", y=metric_choice, title=f"Teams by {metric_choice.replace('_', ' ')}",
        )
        fig3.update_layout(height=420, plot_bgcolor="white", xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption(
    "Cricket Performance Analysis & Prediction System · Python for Data Science "
    "Team Mini Project · Model by teammate, Application by you 🚀"
)

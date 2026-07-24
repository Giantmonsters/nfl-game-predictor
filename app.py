import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ✅ Page config
st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="centered"
)

# ✅ NFL colour styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0a0a0a;
        color: white;
    }
    .stButton > button {
        background-color: #D50A0A;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #013369;
    }
    .stSelectbox label {
        color: white;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Title
st.title("🏈 NFL Game Outcome Predictor")
st.markdown("Predict the outcome of any NFL matchup using machine learning trained on historical data since 1990.")

# ✅ Load and train model
@st.cache_resource
def load_model():
    scores = pd.read_csv('data/spreadspoke_scores.csv')
    scores = scores[(scores['score_home'] > 0) | (scores['score_away'] > 0)]
    scores['home_win'] = (scores['score_home'] > scores['score_away']).astype(int)
    scores = scores.sort_values('schedule_season')
    scores = scores[scores['schedule_season'] >= 1990].copy()

    def get_team_stats(df, team, before_season):
        home_games = df[(df['team_home'] == team) & (df['schedule_season'] < before_season)]
        away_games = df[(df['team_away'] == team) & (df['schedule_season'] < before_season)]
        home_wins = (home_games['score_home'] > home_games['score_away']).sum()
        away_wins = (away_games['score_away'] > away_games['score_home']).sum()
        total_games = len(home_games) + len(away_games)
        if total_games == 0:
            return 0.5, 22.0, 20.0
        win_rate = (home_wins + away_wins) / total_games
        avg_scored = pd.concat([home_games['score_home'], away_games['score_away']]).mean()
        avg_conceded = pd.concat([home_games['score_away'], away_games['score_home']]).mean()
        return win_rate, avg_scored, avg_conceded

    game_data = []
    for _, row in scores.iterrows():
        season = row['schedule_season']
        home_wr, home_scored, home_conceded = get_team_stats(scores, row['team_home'], season)
        away_wr, away_scored, away_conceded = get_team_stats(scores, row['team_away'], season)
        game_data.append({
            'home_win_rate': home_wr,
            'away_win_rate': away_wr,
            'home_avg_scored': home_scored,
            'away_avg_scored': away_scored,
            'home_avg_conceded': home_conceded,
            'away_avg_conceded': away_conceded,
            'home_win': row['home_win']
        })

    df_features = pd.DataFrame(game_data).dropna()
    X = df_features.drop('home_win', axis=1)
    y = df_features['home_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    acc = accuracy_score(y_test, xgb.predict(X_test))

    return xgb, scores, get_team_stats, acc

# ✅ Load model
with st.spinner('Loading model... this may take a minute on first load!'):
    xgb, scores, get_team_stats, accuracy = load_model()

st.success(f"Model loaded! Accuracy: {accuracy:.1%}")

# ✅ Get list of teams
all_teams = sorted(list(set(scores['team_home'].unique()) | set(scores['team_away'].unique())))

# ✅ Team selection
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select home team", all_teams, index=all_teams.index("Kansas City Chiefs") if "Kansas City Chiefs" in all_teams else 0)

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox("Select away team", all_teams, index=all_teams.index("Philadelphia Eagles") if "Philadelphia Eagles" in all_teams else 1)

# ✅ Predict button
st.markdown("---")
if st.button("🔮 Predict Game Outcome", use_container_width=True):
    if home_team == away_team:
        st.error("Please select two different teams!")
    else:
        home_wr, home_scored, home_conceded = get_team_stats(scores, home_team, 2026)
        away_wr, away_scored, away_conceded = get_team_stats(scores, away_team, 2026)

        game_features = [[home_wr, away_wr, home_scored, away_scored, home_conceded, away_conceded]]
        prob = xgb.predict_proba(game_features)[0]
        home_prob = prob[1]
        away_prob = prob[0]

        # ✅ Result
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=f"🏠 {home_team}", value=f"{home_prob:.1%}")
        with col2:
            st.metric(label="VS", value="")
        with col3:
            st.metric(label=f"✈️ {away_team}", value=f"{away_prob:.1%}")

        # ✅ Winner
        if home_prob > away_prob:
            st.success(f"🏆 Predicted Winner: **{home_team}** ({home_prob:.1%} probability)")
        else:
            st.success(f"🏆 Predicted Winner: **{away_team}** ({away_prob:.1%} probability)")

        # ✅ Team stats comparison
        st.markdown("---")
        st.subheader("📈 Team Stats Comparison")

        stats_df = pd.DataFrame({
            'Stat': ['Win Rate', 'Avg Points Scored', 'Avg Points Conceded'],
            home_team: [f"{home_wr:.1%}", f"{home_scored:.1f}", f"{home_conceded:.1f}"],
            away_team: [f"{away_wr:.1%}", f"{away_scored:.1f}", f"{away_conceded:.1f}"]
        })
        st.table(stats_df)

# ✅ Footer
st.markdown("---")
st.caption("Built by NFLNerd | Trained on NFL data from 1990-2025 | Model accuracy: 58.4%")
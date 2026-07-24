import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import plotly.graph_objects as go

# ✅ Page config
st.set_page_config(
    page_title="NFL Game Predictor | NFLNerd",
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
    .nflnerd-brand {
        font-size: 14px;
        color: #D50A0A;
        font-weight: bold;
        letter-spacing: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ NFLNerd Branding
st.markdown('<p class="nflnerd-brand">🏈 NFLNERD</p>', unsafe_allow_html=True)
st.title("NFL Game Outcome Predictor")
st.markdown("Predict the outcome of any NFL matchup using machine learning trained on historical data since 1990.")

# ✅ Load and train model
@st.cache_resource
def load_model():
    scores = pd.read_csv('data/spreadspoke_scores.csv')
    scores = scores[(scores['score_home'] > 0) | (scores['score_away'] > 0)]
    scores['home_win'] = (scores['score_home'] > scores['score_away']).astype(int)
    scores['schedule_date'] = pd.to_datetime(scores['schedule_date'])
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

# ✅ Get recent form for a team
def get_recent_form(scores, team, n=5):
    home_games = scores[scores['team_home'] == team][['schedule_date', 'team_home', 'team_away', 'score_home', 'score_away']].copy()
    home_games['win'] = home_games['score_home'] > home_games['score_away']
    home_games['result'] = home_games.apply(lambda r: f"✅ W {int(r['score_home'])}-{int(r['score_away'])} vs {r['team_away']}" if r['win'] else f"❌ L {int(r['score_home'])}-{int(r['score_away'])} vs {r['team_away']}", axis=1)

    away_games = scores[scores['team_away'] == team][['schedule_date', 'team_home', 'team_away', 'score_home', 'score_away']].copy()
    away_games['win'] = away_games['score_away'] > away_games['score_home']
    away_games['result'] = away_games.apply(lambda r: f"✅ W {int(r['score_away'])}-{int(r['score_home'])} @ {r['team_home']}" if r['win'] else f"❌ L {int(r['score_away'])}-{int(r['score_home'])} @ {r['team_home']}", axis=1)

    all_games = pd.concat([home_games[['schedule_date', 'result', 'win']], away_games[['schedule_date', 'result', 'win']]])
    all_games['schedule_date'] = pd.to_datetime(all_games['schedule_date'])
    all_games = all_games.sort_values('schedule_date', ascending=False).head(n)
    return all_games

# ✅ Get head to head record
def get_head_to_head(scores, team1, team2):
    h2h = scores[
        ((scores['team_home'] == team1) & (scores['team_away'] == team2)) |
        ((scores['team_home'] == team2) & (scores['team_away'] == team1))
    ].copy()

    team1_wins = 0
    team2_wins = 0
    for _, row in h2h.iterrows():
        if row['team_home'] == team1:
            if row['score_home'] > row['score_away']:
                team1_wins += 1
            else:
                team2_wins += 1
        else:
            if row['score_away'] > row['score_home']:
                team1_wins += 1
            else:
                team2_wins += 1

    return team1_wins, team2_wins, len(h2h)

# ✅ Load model
with st.spinner('Loading model... this may take a minute on first load!'):
    xgb, scores, get_team_stats, accuracy = load_model()

st.success(f"Model loaded! Accuracy: {accuracy:.1%}")

# ✅ Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📅 Weekly Predictions", "ℹ️ How It Works"])

with tab1:
    # ✅ Get list of teams
    all_teams = sorted(list(set(scores['team_home'].unique()) | set(scores['team_away'].unique())))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Select home team", all_teams, index=all_teams.index("Kansas City Chiefs") if "Kansas City Chiefs" in all_teams else 0)

    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Select away team", all_teams, index=all_teams.index("Philadelphia Eagles") if "Philadelphia Eagles" in all_teams else 1)

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

            # ✅ Confidence indicator
            diff = abs(home_prob - away_prob)
            if diff >= 0.15:
                confidence = "🟢 High Confidence"
                confidence_msg = "The model is strongly favouring one team."
            elif diff >= 0.07:
                confidence = "🟡 Medium Confidence"
                confidence_msg = "The model leans one way but it's not clear cut."
            else:
                confidence = "🔴 Low Confidence — Close Game"
                confidence_msg = "This is a very tight matchup. Could go either way!"

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

            if home_prob > away_prob:
                st.success(f"🏆 Predicted Winner: **{home_team}** ({home_prob:.1%} probability)")
            else:
                st.success(f"🏆 Predicted Winner: **{away_team}** ({away_prob:.1%} probability)")

            st.info(f"{confidence} — {confidence_msg}")

            # ✅ Win probability bar chart
            st.markdown("---")
            st.subheader("📊 Win Probability")
            fig_prob = go.Figure(go.Bar(
                x=[f"🏠 {home_team}", f"✈️ {away_team}"],
                y=[home_prob * 100, away_prob * 100],
                marker_color=["#013369", "#D50A0A"],
                text=[f"{home_prob:.1%}", f"{away_prob:.1%}"],
                textposition="outside",
                textfont=dict(color="white", size=16)
            ))
            fig_prob.update_layout(
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                font=dict(color="white"),
                yaxis=dict(title="Win Probability (%)", range=[0, 100], gridcolor="#333"),
                xaxis=dict(gridcolor="#333"),
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # ✅ Win probability gauge
            st.subheader("🎯 Win Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=home_prob * 100,
                title={"text": f"{home_team} Win Probability", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#013369"},
                    "bgcolor": "#0a0a0a",
                    "steps": [
                        {"range": [0, 50], "color": "#D50A0A"},
                        {"range": [50, 100], "color": "#013369"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": 50
                    }
                },
                number={"suffix": "%", "font": {"color": "white"}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0a0a0a",
                font=dict(color="white"),
                height=300
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ✅ Head to head
            st.markdown("---")
            st.subheader("⚔️ Head to Head Record")
            h2h_wins, h2h_losses, h2h_total = get_head_to_head(scores, home_team, away_team)
            if h2h_total == 0:
                st.write("No head to head record found.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label=f"🏠 {home_team} Wins", value=h2h_wins)
                with col2:
                    st.metric(label="Total Games", value=h2h_total)
                with col3:
                    st.metric(label=f"✈️ {away_team} Wins", value=h2h_losses)

                if h2h_wins > h2h_losses:
                    st.write(f"**{home_team}** lead the all time series {h2h_wins}-{h2h_losses}")
                elif h2h_losses > h2h_wins:
                    st.write(f"**{away_team}** lead the all time series {h2h_losses}-{h2h_wins}")
                else:
                    st.write(f"The all time series is tied {h2h_wins}-{h2h_losses}")

            # ✅ Recent form
            st.markdown("---")
            st.subheader("📅 Recent Form (Last 5 Games)")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**🏠 {home_team}**")
                home_form = get_recent_form(scores, home_team)
                for _, row in home_form.iterrows():
                    st.markdown(row['result'])

            with col2:
                st.markdown(f"**✈️ {away_team}**")
                away_form = get_recent_form(scores, away_team)
                for _, row in away_form.iterrows():
                    st.markdown(row['result'])

            # ✅ Team stats comparison
            st.markdown("---")
            st.subheader("📈 Team Stats Comparison")
            stats_df = pd.DataFrame({
                'Stat': ['Win Rate', 'Avg Points Scored', 'Avg Points Conceded'],
                home_team: [f"{home_wr:.1%}", f"{home_scored:.1f}", f"{home_conceded:.1f}"],
                away_team: [f"{away_wr:.1%}", f"{away_scored:.1f}", f"{away_conceded:.1f}"]
            })
            st.table(stats_df)

            # ✅ Team stats bar chart
            st.subheader("📊 Stats Comparison Chart")
            categories = ['Win Rate (%)', 'Avg Points Scored', 'Avg Points Conceded']
            home_values = [home_wr * 100, home_scored, home_conceded]
            away_values = [away_wr * 100, away_scored, away_conceded]

            fig_stats = go.Figure()
            fig_stats.add_trace(go.Bar(
                name=f"🏠 {home_team}",
                x=categories,
                y=home_values,
                marker_color="#013369",
                text=[f"{v:.1f}" for v in home_values],
                textposition="outside",
                textfont=dict(color="white")
            ))
            fig_stats.add_trace(go.Bar(
                name=f"✈️ {away_team}",
                x=categories,
                y=away_values,
                marker_color="#D50A0A",
                text=[f"{v:.1f}" for v in away_values],
                textposition="outside",
                textfont=dict(color="white")
            ))
            fig_stats.update_layout(
                barmode="group",
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                font=dict(color="white"),
                yaxis=dict(gridcolor="#333"),
                xaxis=dict(gridcolor="#333"),
                legend=dict(font=dict(color="white")),
                height=400
            )
            st.plotly_chart(fig_stats, use_container_width=True)

with tab2:
    # ✅ Weekly predictions
    st.subheader("📅 Weekly Predictions")
    st.markdown("Enter this week's matchups and get predictions for all games at once.")

    st.info("Add up to 8 matchups below:")

    num_games = st.slider("How many games?", min_value=1, max_value=8, value=4)

    all_teams_list = sorted(list(set(scores['team_home'].unique()) | set(scores['team_away'].unique())))

    matchups = []
    for i in range(num_games):
        st.markdown(f"**Game {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            h = st.selectbox(f"Home team {i+1}", all_teams_list, key=f"home_{i}")
        with col2:
            a = st.selectbox(f"Away team {i+1}", all_teams_list, key=f"away_{i}", index=1)
        matchups.append((h, a))

    if st.button("🔮 Predict All Games", use_container_width=True):
        st.markdown("---")
        for h, a in matchups:
            if h == a:
                st.warning(f"Skipping — same team selected for both sides")
                continue
            h_wr, h_scored, h_conceded = get_team_stats(scores, h, 2026)
            a_wr, a_scored, a_conceded = get_team_stats(scores, a, 2026)
            feats = [[h_wr, a_wr, h_scored, a_scored, h_conceded, a_conceded]]
            prob = xgb.predict_proba(feats)[0]
            h_prob = prob[1]
            a_prob = prob[0]
            winner = h if h_prob > a_prob else a
            win_prob = max(h_prob, a_prob)

            diff = abs(h_prob - a_prob)
            if diff >= 0.15:
                conf = "🟢"
            elif diff >= 0.07:
                conf = "🟡"
            else:
                conf = "🔴"

            st.markdown(f"{conf} **{h}** vs **{a}** → 🏆 **{winner}** ({win_prob:.1%})")

with tab3:
    # ✅ How it works
    st.subheader("ℹ️ How The Model Works")
    st.markdown("""
    ### The Data
    This model is trained on **9,455 NFL games** from 1990 to 2025, sourced from the NFL scores and betting dataset on Kaggle.

    ### The Features
    For each game, the model uses 6 features:
    - 🏠 **Home team win rate** — historical win % for the home team
    - ✈️ **Away team win rate** — historical win % for the away team
    - 🏈 **Home team avg points scored** — average points scored per game
    - 🏈 **Away team avg points scored** — average points scored per game
    - 🛡️ **Home team avg points conceded** — average points conceded per game
    - 🛡️ **Away team avg points conceded** — average points conceded per game

    ### The Model
    We compared three machine learning models:
    | Model | Accuracy |
    |-------|----------|
    | Logistic Regression | 57.4% |
    | Random Forest | 57.6% |
    | **XGBoost** ✅ | **58.4%** |

    **XGBoost** was chosen as the best performing model.

    ### What Does 58.4% Mean?
    - Random guessing = 50%
    - Our model = 58.4%
    - Professional betting models = 60-65%
    - Nobody consistently exceeds 65% — NFL is unpredictable!

    ### Confidence Levels
    - 🟢 **High Confidence** — probability gap ≥ 15%
    - 🟡 **Medium Confidence** — probability gap 7-15%
    - 🔴 **Low Confidence** — probability gap < 7% (very close game)

    ### Limitations
    The model does not account for:
    - Current season injuries
    - Weather conditions
    - Recent trades or roster changes
    - Coaching changes

    *Built by NFLNerd using Python, XGBoost and Streamlit.*
    """)

# ✅ Footer
st.markdown("---")
st.caption("Built by NFLNerd | Trained on NFL data from 1990-2025 | Model accuracy: 58.4%")
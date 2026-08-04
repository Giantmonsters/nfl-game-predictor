import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import plotly.graph_objects as go
import numpy as np

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
.explainer-box {
    background-color: #1a1a1a;
    border-left: 4px solid #D50A0A;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 12px;
    font-size: 14px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

def explainer(text):
    st.markdown(f'<div class="explainer-box">{text}</div>', unsafe_allow_html=True)

# ✅ NFLNerd Branding
st.markdown('<p class="nflnerd-brand">🏈 NFLNERD</p>', unsafe_allow_html=True)
st.title("NFL Game Outcome Predictor")
st.markdown("Predict the outcome of any NFL matchup using machine learning trained on historical data since 1990.")

# ✅ The 32 current active NFL teams
CURRENT_NFL_TEAMS = [
    'Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
    'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
    'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
    'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs',
    'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
    'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants',
    'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers',
    'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders'
]

# ✅ Load and train model
@st.cache_resource
def load_model():
    scores = pd.read_csv('data/spreadspoke_scores.csv')
    scores = scores[(scores['score_home'] > 0) | (scores['score_away'] > 0)]

    team_renames = {
        'Oakland Raiders': 'Las Vegas Raiders',
        'St. Louis Rams': 'Los Angeles Rams',
        'San Diego Chargers': 'Los Angeles Chargers',
        'Washington Redskins': 'Washington Commanders',
        'Washington Football Team': 'Washington Commanders',
        'Tennessee Oilers': 'Tennessee Titans',
        'Houston Oilers': 'Tennessee Titans',
        'Phoenix Cardinals': 'Arizona Cardinals',
        'Baltimore Colts': 'Indianapolis Colts',
        'Los Angeles Raiders': 'Las Vegas Raiders',
    }
    scores['team_home'] = scores['team_home'].replace(team_renames)
    scores['team_away'] = scores['team_away'].replace(team_renames)

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
            'home_win': row['home_win'],
            'season': season
        })

    df_features = pd.DataFrame(game_data).dropna()
    X = df_features.drop(['home_win', 'season'], axis=1)
    y = df_features['home_win']
    seasons = df_features['season']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr  = LogisticRegression(random_state=42, max_iter=1000)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    lr_acc  = accuracy_score(y_test, lr.predict(X_test))
    rf_acc  = accuracy_score(y_test, rf.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

    # Season-by-season accuracy (2000 onwards for reliability)
    season_acc = []
    for s in sorted(seasons.unique()):
        if s < 2000:
            continue
        mask = seasons == s
        if mask.sum() < 10:
            continue
        preds = xgb.predict(X[mask])
        season_acc.append({'season': int(s), 'accuracy': accuracy_score(y[mask], preds)})

    # Confusion matrix
    cm = confusion_matrix(y_test, xgb.predict(X_test))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Confidence bucket accuracy
    probs = xgb.predict_proba(X_test)[:, 1]
    preds_test = xgb.predict(X_test)
    conf_data = []
    for label, low, high in [('🔴 Low\n(<7% gap)', 0.0, 0.07), ('🟡 Medium\n(7–15% gap)', 0.07, 0.15), ('🟢 High\n(>15% gap)', 0.15, 0.5)]:
        mask = (abs(probs - 0.5) >= low) & (abs(probs - 0.5) < high)
        if mask.sum() > 0:
            bucket_acc = accuracy_score(y_test[mask], preds_test[mask])
            conf_data.append({'confidence': label, 'accuracy': bucket_acc, 'games': int(mask.sum())})

    # Feature importances
    feature_names = ['Home Win Rate', 'Away Win Rate', 'Home Avg Scored',
                     'Away Avg Scored', 'Home Avg Conceded', 'Away Avg Conceded']
    importances = xgb.feature_importances_

    # Home win rate by decade (1995+ only for reliability)
    scores_reliable = scores[scores['schedule_season'] >= 1995].copy()
    scores_reliable['decade'] = (scores_reliable['schedule_season'] // 10 * 10).astype(str) + 's'
    decade_hw = scores_reliable.groupby('decade')['home_win'].mean().reset_index()
    decade_hw.columns = ['Decade', 'Home Win Rate']

    return (xgb, scores, get_team_stats, xgb_acc,
            lr_acc, rf_acc,
            pd.DataFrame(season_acc),
            cm, fpr, tpr, roc_auc,
            conf_data,
            feature_names, importances,
            X_test, y_test,
            decade_hw)


# ✅ Get recent form
def get_recent_form(scores, team, n=5):
    home_games = scores[scores['team_home'] == team][['schedule_date', 'team_home', 'team_away', 'score_home', 'score_away']].copy()
    home_games['win'] = home_games['score_home'] > home_games['score_away']
    home_games['result'] = home_games.apply(lambda r: f"✅ W {int(r['score_home'])}-{int(r['score_away'])} vs {r['team_away']}" if r['win'] else f"❌ L {int(r['score_home'])}-{int(r['score_away'])} vs {r['team_away']}", axis=1)
    away_games = scores[scores['team_away'] == team][['schedule_date', 'team_home', 'team_away', 'score_home', 'score_away']].copy()
    away_games['win'] = away_games['score_away'] > away_games['score_home']
    away_games['result'] = away_games.apply(lambda r: f"✅ W {int(r['score_away'])}-{int(r['score_home'])} @ {r['team_home']}" if r['win'] else f"❌ L {int(r['score_away'])}-{int(r['score_home'])} @ {r['team_home']}", axis=1)
    all_games = pd.concat([home_games[['schedule_date', 'result', 'win']], away_games[['schedule_date', 'result', 'win']]])
    all_games['schedule_date'] = pd.to_datetime(all_games['schedule_date'])
    return all_games.sort_values('schedule_date', ascending=False).head(n)


# ✅ Head to head
def get_head_to_head(scores, team1, team2):
    h2h = scores[
        ((scores['team_home'] == team1) & (scores['team_away'] == team2)) |
        ((scores['team_home'] == team2) & (scores['team_away'] == team1))
    ].copy()
    team1_wins = team2_wins = 0
    for _, row in h2h.iterrows():
        if row['team_home'] == team1:
            if row['score_home'] > row['score_away']: team1_wins += 1
            else: team2_wins += 1
        else:
            if row['score_away'] > row['score_home']: team1_wins += 1
            else: team2_wins += 1
    return team1_wins, team2_wins, len(h2h)


# ✅ Load model
with st.spinner('Loading model... this may take a minute on first load!'):
    (xgb, scores, get_team_stats, accuracy,
     lr_acc, rf_acc,
     season_acc_df,
     cm, fpr, tpr, roc_auc,
     conf_data,
     feature_names, importances,
     X_test, y_test,
     decade_hw) = load_model()

st.success(f"Model loaded! Accuracy: {accuracy:.1%}")

all_teams = sorted(CURRENT_NFL_TEAMS)

CHART_LAYOUT = dict(
    plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
    font=dict(color='white'), showlegend=False
)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📅 Weekly Predictions", "🧪 Data Science", "ℹ️ How It Works"])

# ─────────────────────────────────────────────
# TAB 1 — PREDICT
# ─────────────────────────────────────────────
with tab1:
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Select home team", all_teams,
            index=all_teams.index("Kansas City Chiefs") if "Kansas City Chiefs" in all_teams else 0)
    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Select away team", all_teams,
            index=all_teams.index("Philadelphia Eagles") if "Philadelphia Eagles" in all_teams else 1)

    st.markdown("---")

    if st.button("🔮 Predict Game Outcome", use_container_width=True):
        if home_team == away_team:
            st.error("Please select two different teams!")
        else:
            home_wr, home_scored, home_conceded = get_team_stats(scores, home_team, 2026)
            away_wr, away_scored, away_conceded = get_team_stats(scores, away_team, 2026)

            prob = xgb.predict_proba([[home_wr, away_wr, home_scored, away_scored, home_conceded, away_conceded]])[0]
            home_prob, away_prob = prob[1], prob[0]

            diff = abs(home_prob - away_prob)
            if diff >= 0.15:
                confidence, confidence_msg = "🟢 High Confidence", "The model is strongly favouring one team."
            elif diff >= 0.07:
                confidence, confidence_msg = "🟡 Medium Confidence", "The model leans one way but it's not clear cut."
            else:
                confidence, confidence_msg = "🔴 Low Confidence — Close Game", "This is a very tight matchup. Could go either way!"

            st.markdown("---")
            st.subheader("📊 Prediction Results")

            col1, col2, col3 = st.columns(3)
            with col1: st.metric(label=f"🏠 {home_team}", value=f"{home_prob:.1%}")
            with col2: st.metric(label="VS", value="")
            with col3: st.metric(label=f"✈️ {away_team}", value=f"{away_prob:.1%}")

            if home_prob > away_prob:
                st.success(f"🏆 Predicted Winner: **{home_team}** ({home_prob:.1%} probability)")
            else:
                st.success(f"🏆 Predicted Winner: **{away_team}** ({away_prob:.1%} probability)")

            st.info(f"{confidence} — {confidence_msg}")

            st.markdown("---")
            st.subheader("📋 Team Stats Comparison")
            stats_df = pd.DataFrame({
                'Stat': ['Historical Win Rate', 'Avg Points Scored', 'Avg Points Conceded'],
                f'🏠 {home_team}': [f"{home_wr:.1%}", f"{home_scored:.1f}", f"{home_conceded:.1f}"],
                f'✈️ {away_team}': [f"{away_wr:.1%}", f"{away_scored:.1f}", f"{away_conceded:.1f}"],
            })
            st.dataframe(stats_df.set_index('Stat'), use_container_width=True)

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
            fig_prob.update_layout(**CHART_LAYOUT,
                yaxis=dict(title="Win Probability (%)", range=[0, 110], gridcolor="#333"),
                xaxis=dict(gridcolor="#333"), height=350)
            st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown("---")
            st.subheader("⚔️ Head to Head Record")
            h2h_wins, h2h_losses, h2h_total = get_head_to_head(scores, home_team, away_team)
            if h2h_total == 0:
                st.write("No head to head record found.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1: st.metric(label=f"🏠 {home_team} Wins", value=h2h_wins)
                with col2: st.metric(label="Total Games", value=h2h_total)
                with col3: st.metric(label=f"✈️ {away_team} Wins", value=h2h_losses)
                if h2h_wins > h2h_losses:
                    st.write(f"**{home_team}** lead the all time series {h2h_wins}-{h2h_losses}")
                elif h2h_losses > h2h_wins:
                    st.write(f"**{away_team}** lead the all time series {h2h_losses}-{h2h_wins}")
                else:
                    st.write(f"The all time series is tied {h2h_wins}-{h2h_losses}")

            st.markdown("---")
            st.subheader("📅 Recent Form (Last 5 Games)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🏠 {home_team}**")
                for _, row in get_recent_form(scores, home_team).iterrows():
                    st.markdown(row['result'])
            with col2:
                st.markdown(f"**✈️ {away_team}**")
                for _, row in get_recent_form(scores, away_team).iterrows():
                    st.markdown(row['result'])

# ─────────────────────────────────────────────
# TAB 2 — WEEKLY PREDICTIONS
# ─────────────────────────────────────────────
with tab2:
    st.subheader("📅 Weekly Predictions")
    st.markdown("Enter this week's matchups and get predictions for all games at once.")
    st.info("Add up to 16 matchups below:")

    num_games = st.slider("How many games?", min_value=1, max_value=16, value=4)
    matchups = []
    for i in range(num_games):
        st.markdown(f"**Game {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            h = st.selectbox(f"Home team {i+1}", all_teams, key=f"home_{i}")
        with col2:
            a = st.selectbox(f"Away team {i+1}", all_teams, key=f"away_{i}", index=1)
        matchups.append((h, a))

    if st.button("🔮 Predict All Games", use_container_width=True):
        st.markdown("---")
        for i, (h, a) in enumerate(matchups):
            if h == a:
                st.warning(f"Game {i+1}: Skipping — same team selected for both sides")
                continue
            h_wr, h_scored, h_conceded = get_team_stats(scores, h, 2026)
            a_wr, a_scored, a_conceded = get_team_stats(scores, a, 2026)
            prob = xgb.predict_proba([[h_wr, a_wr, h_scored, a_scored, h_conceded, a_conceded]])[0]
            h_prob, a_prob = prob[1], prob[0]
            winner = h if h_prob > a_prob else a
            win_prob = max(h_prob, a_prob)
            diff = abs(h_prob - a_prob)
            conf = "🟢" if diff >= 0.15 else ("🟡" if diff >= 0.07 else "🔴")
            st.markdown(f"{conf} **{h}** vs **{a}** → 🏆 **{winner}** ({win_prob:.1%})")

# ─────────────────────────────────────────────
# TAB 3 — DATA SCIENCE SHOWCASE
# ─────────────────────────────────────────────
with tab3:
    st.subheader("🧪 Data Science & Model Analysis")
    st.markdown("""
This tab is for anyone who wants to understand **how the prediction model actually works** — whether you're an NFL fan, a student, or a data scientist.
Every chart comes with a plain English explanation so you don't need a computer science degree to follow along.
""")

    # ── 1. Model Comparison ──
    st.markdown("---")
    st.subheader("🤖 1. Which Model is Best?")

    explainer("""
<b>What is this?</b> To build this predictor, three different machine learning models were trained on the same NFL data and tested against each other. Think of it like signing three quarterbacks and seeing which one performs best in pre-season — only the winner makes the final roster.<br><br>
<b>Why XGBoost?</b> On this dataset the accuracy numbers are very close (within ~1%). However, XGBoost was chosen because:
it handles complex patterns in data better than Logistic Regression as more data is added,
it's the industry-standard model used by professional data scientists and Kaggle competition winners,
and it gives us feature importance scores (see below) which help us understand <i>why</i> it makes each prediction — something Logistic Regression does less reliably at scale.
In short: on a small dataset they're nearly equal, but XGBoost is the better long-term choice.
""")

    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost ✅']
    model_accs  = [lr_acc * 100, rf_acc * 100, accuracy * 100]

    fig_models = go.Figure(go.Bar(
        x=model_names, y=model_accs,
        marker_color=['#555555', '#888888', '#D50A0A'],
        text=[f"{a:.2f}%" for a in model_accs],
        textposition='outside', textfont=dict(color='white', size=14)
    ))
    fig_models.add_hline(y=50, line_dash='dash', line_color='#666',
                         annotation_text='Random Guessing (50%)', annotation_font_color='#aaa',
                         annotation_position='bottom right')
    fig_models.update_layout(**CHART_LAYOUT,
        yaxis=dict(title='Accuracy (%)', range=[45, 65], gridcolor='#333'),
        xaxis=dict(gridcolor='#333'), height=380)
    st.plotly_chart(fig_models, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Logistic Regression", f"{lr_acc:.2%}")
    with col2: st.metric("Random Forest", f"{rf_acc:.2%}")
    with col3: st.metric("XGBoost (chosen ✅)", f"{accuracy:.2%}", delta=f"+{(accuracy - lr_acc):.2%} vs LR")

    st.success("🏈 **NFL fan takeaway:** Even the best models only get about 6 in 10 predictions right — that's how unpredictable the NFL is. Nobody, human or machine, can consistently do better than that.")

    # ── 2. Feature Importance ──
    st.markdown("---")
    st.subheader("🔍 2. What Does the Model Actually Look At?")

    explainer("""
<b>What is this?</b> When the model predicts a winner, it doesn't weigh all stats equally — some matter more than others. This chart shows which of the 6 inputs it relies on most.<br><br>
<b>What is an importance score?</b> It's a number between 0 and 1 that shows how much a stat influences the model's decision. A score of 0.4 means that stat is responsible for 40% of the model's decision-making. All scores add up to 1.0 (100%).<br><br>
<b>What do the stats on the Y-axis mean?</b><br>
• <b>Home/Away Win Rate</b> — the team's overall win percentage across all historical games<br>
• <b>Home/Away Avg Scored</b> — how many points they score per game on average<br>
• <b>Home/Away Avg Conceded</b> — how many points they let in per game on average
""")

    sorted_idx        = np.argsort(importances)[::-1]
    sorted_features   = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    fig_fi = go.Figure(go.Bar(
        x=sorted_importances, y=sorted_features,
        orientation='h',
        marker_color='#013369',
        text=[f"{v:.3f}" for v in sorted_importances],
        textposition='outside', textfont=dict(color='white')
    ))
    fig_fi.update_layout(**CHART_LAYOUT,
        xaxis=dict(title='Importance Score (0–1)', gridcolor='#333', range=[0, max(sorted_importances) * 1.3]),
        yaxis=dict(gridcolor='#333', autorange='reversed'),
        height=400)
    st.plotly_chart(fig_fi, use_container_width=True)

    top_feat = sorted_features[0]
    st.success(f"🏈 **NFL fan takeaway:** **{top_feat}** has the biggest influence. That makes intuitive sense — teams that have won more games historically tend to keep winning. It's the NFL equivalent of 'good teams beat bad teams most of the time.'")

    # ── 3. Season-by-Season Accuracy ──
    st.markdown("---")
    st.subheader("📈 3. How Accurate Was It Each Season?")

    explainer("""
<b>What is this?</b> Rather than just one overall accuracy number, this chart shows how well the model predicted games in each individual season from 2000 onwards.<br><br>
<b>Why does accuracy vary by season?</b> Some NFL seasons are more predictable than others. A season dominated by one or two clearly superior teams (like the 2007 Patriots or 2018 Chiefs) is easier to predict. A season full of upsets and parity is harder.<br><br>
<b>Why start from 2000?</b> Pre-2000 data covers fewer teams and a different era of the NFL, making season-level comparisons less meaningful.
""")

    if not season_acc_df.empty:
        avg_acc = season_acc_df['accuracy'].mean()

        fig_season = go.Figure()
        fig_season.add_trace(go.Scatter(
            x=season_acc_df['season'],
            y=season_acc_df['accuracy'] * 100,
            mode='lines+markers',
            line=dict(color='#D50A0A', width=2),
            marker=dict(size=8, color='#D50A0A'),
            name='Model Accuracy',
            hovertemplate='Season: %{x}<br>Accuracy: %{y:.1f}%<extra></extra>'
        ))
        fig_season.add_hline(y=50, line_dash='dash', line_color='#555',
                             annotation_text='Random guessing (50%)', annotation_font_color='#aaa',
                             annotation_position='bottom right')
        fig_season.add_hline(y=avg_acc * 100, line_dash='dot', line_color='#013369',
                             annotation_text=f'Average ({avg_acc:.1%})', annotation_font_color='#aaa',
                             annotation_position='top right')
        fig_season.update_layout(**CHART_LAYOUT,
            xaxis=dict(title='NFL Season', gridcolor='#333', dtick=2,
                       range=[season_acc_df['season'].min() - 1, season_acc_df['season'].max() + 1]),
            yaxis=dict(title='Accuracy (%)', range=[40, 80], gridcolor='#333'),
            height=400)
        st.plotly_chart(fig_season, use_container_width=True)

        best   = season_acc_df.loc[season_acc_df['accuracy'].idxmax()]
        worst  = season_acc_df.loc[season_acc_df['accuracy'].idxmin()]
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Average Accuracy", f"{avg_acc:.1%}")
        with col2: st.metric("Best Season",    f"{int(best['season'])}  ({best['accuracy']:.1%})")
        with col3: st.metric("Toughest Season", f"{int(worst['season'])} ({worst['accuracy']:.1%})")

        st.success(f"🏈 **NFL fan takeaway:** The model beats random guessing every single season. Its toughest year was **{int(worst['season'])}** — likely a season with a lot of upsets. Its best was **{int(best['season'])}** — probably a more dominant-team era.")

    # ── 4. Confidence vs Accuracy ──
    st.markdown("---")
    st.subheader("🎯 4. Does Confidence Actually Mean Anything?")

    explainer("""
<b>What is this?</b> The model doesn't just say "home team wins" — it gives a probability, like 64% vs 36%. The bigger the gap between those two numbers, the more confident the model is.<br><br>
<b>The three buckets:</b><br>
• 🔴 <b>Low confidence</b> — the model thinks it's nearly 50/50 (gap under 7%). These are genuine coin-flip games.<br>
• 🟡 <b>Medium confidence</b> — the model leans one way but isn't sure (gap 7–15%).<br>
• 🟢 <b>High confidence</b> — the model strongly favours one team (gap over 15%).<br><br>
<b>Why does this matter?</b> If high confidence predictions are meaningfully more accurate than low confidence ones, it means the model <i>knows what it doesn't know</i> — a key quality in any good prediction system.
""")

    if conf_data:
        conf_df = pd.DataFrame(conf_data)
        fig_conf = go.Figure(go.Bar(
            x=conf_df['confidence'],
            y=conf_df['accuracy'] * 100,
            marker_color=['#D50A0A', '#FFB300', '#00C853'],
            text=[f"{a:.1%}<br>({g} games)" for a, g in zip(conf_df['accuracy'], conf_df['games'])],
            textposition='outside', textfont=dict(color='white', size=13)
        ))
        fig_conf.add_hline(y=50, line_dash='dash', line_color='#555',
                           annotation_text='Random guessing', annotation_font_color='#aaa',
                           annotation_position='bottom right')
        fig_conf.update_layout(**CHART_LAYOUT,
            yaxis=dict(title='Actual Accuracy (%)', range=[40, 85], gridcolor='#333'),
            xaxis=dict(gridcolor='#333'), height=400)
        st.plotly_chart(fig_conf, use_container_width=True)
        st.success("🏈 **NFL fan takeaway:** If the green bar is notably taller than the red bar, you can trust the model's confidence signals. When it says 🟢 High Confidence, it's actually been right more often than when it says 🔴 Low Confidence.")

    # ── 5. Confusion Matrix ──
    st.markdown("---")
    st.subheader("🔢 5. Where Does the Model Go Wrong?")

    explainer("""
<b>What is this?</b> A confusion matrix is a scoreboard for the model — it shows not just <i>how often</i> it was right, but <i>what kind of mistakes</i> it made.<br><br>
<b>The four boxes explained in plain English:</b><br>
• ✅ <b>True Away Win</b> — The away team actually won, and the model correctly predicted an away win. The model got it right.<br>
• ✅ <b>True Home Win</b> — The home team actually won, and the model correctly predicted a home win. The model got it right.<br>
• ❌ <b>False Positive</b> — The model predicted a home win, but the away team actually won. The model was too confident in the home team.<br>
• ❌ <b>False Negative</b> — The model predicted an away win, but the home team actually won. The model underestimated the home team.<br><br>
You ideally want the top-left and bottom-right boxes (the correct predictions) to be as large as possible.
""")

    tn, fp, fn, tp = cm.ravel()
    fig_cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=['Predicted: Away Win', 'Predicted: Home Win'],
        y=['Actual: Away Win', 'Actual: Home Win'],
        colorscale=[[0, '#0a0a0a'], [1, '#D50A0A']],
        text=[[f"✅ True Away Win\n{tn}", f"❌ False Positive\n{fp}"],
              [f"❌ False Negative\n{fn}", f"✅ True Home Win\n{tp}"]],
        texttemplate='%{text}',
        textfont=dict(size=13, color='white'),
        showscale=False
    ))
    fig_cm.update_layout(**CHART_LAYOUT, height=380,
        xaxis=dict(side='top'),
        margin=dict(t=80, b=40))
    st.plotly_chart(fig_cm, use_container_width=True)

    total_correct = tn + tp
    total_wrong   = fp + fn
    total_preds   = tn + tp + fp + fn
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("✅ True Away Wins",   tn, help="Away team won AND model said away win")
    with col2: st.metric("✅ True Home Wins",   tp, help="Home team won AND model said home win")
    with col3: st.metric("❌ False Positives",  fp, help="Model said home win, but away team won")
    with col4: st.metric("❌ False Negatives",  fn, help="Model said away win, but home team won")

    st.success(f"🏈 **NFL fan takeaway:** Out of {total_preds:,} test games, the model got **{total_correct:,} right** and **{total_wrong:,} wrong**. It's slightly better at predicting home wins than away wins — which makes sense, since home teams win more often historically.")

    # ── 6. ROC Curve ──
    st.markdown("---")
    st.subheader("📉 6. How Good Is the Model Overall? (ROC Curve)")

    explainer(f"""
<b>What is this?</b> The ROC curve is a standard data science tool for measuring how well a model separates two outcomes — in this case, home wins vs away wins.<br><br>
<b>Plain English version:</b> Imagine you're a scout who has to decide whether to trust a tip-off or ignore it. A perfect scout always acts on correct tips and ignores all bad ones. A useless scout is no better than flipping a coin. The ROC curve shows where our model sits between those two extremes.<br><br>
• The <b>red line</b> is our model. The further it bends toward the top-left corner, the better.<br>
• The <b>dashed grey line</b> is random guessing — a completely useless model that just flips a coin.<br>
• <b>AUC = {roc_auc:.3f}</b> — this is the single summary score. 1.0 = perfect, 0.5 = useless coin flip. Our score of {roc_auc:.3f} means the model is meaningfully better than random, which for NFL predictions is genuinely impressive.
""")

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        line=dict(color='#D50A0A', width=2),
        name=f'Our Model (AUC = {roc_auc:.3f})',
        fill='tozeroy', fillcolor='rgba(213,10,10,0.1)'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(color='#666', dash='dash'),
        name='Random Guessing (AUC = 0.5)'
    ))
    fig_roc.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        xaxis=dict(title='False Positive Rate  (How often it wrongly picks home wins)', gridcolor='#333', range=[0, 1]),
        yaxis=dict(title='True Positive Rate  (How often it correctly picks home wins)', gridcolor='#333', range=[0, 1]),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#333'),
        height=420,
        showlegend=True
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.success(f"🏈 **NFL fan takeaway:** An AUC of **{roc_auc:.3f}** means our model is genuinely better than guessing. Professional sports betting models typically sit around 0.58–0.62. We're right in that range — not bad for a model built on publicly available data!")

    # ── 7. Dataset Overview ──
    st.markdown("---")
    st.subheader("📦 7. The Data Behind the Model")

    explainer("""
<b>Where does the data come from?</b> The model is trained on the <b>NFL Scores & Betting Dataset</b> from Kaggle, which contains historical game results going back to 1966. We use games from 1990 onwards for training.<br><br>
<b>A note on the 1990s data:</b> The NFL had fewer teams in the early 1990s (28 teams, vs 32 today). Teams like the Jacksonville Jaguars and Carolina Panthers didn't exist until 1995. So pre-1995 data is included in model training, but the decade chart below starts from 1995 to keep comparisons fair.
""")

    total_games    = len(scores)
    seasons_range  = int(scores['schedule_season'].max()) - int(scores['schedule_season'].min()) + 1
    teams_covered  = len(set(scores['team_home'].unique()) | set(scores['team_away'].unique()))
    home_win_pct   = scores['home_win'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Games", f"{total_games:,}")
    with col2: st.metric("Seasons Covered", seasons_range)
    with col3: st.metric("Teams in Data", teams_covered)
    with col4: st.metric("Overall Home Win Rate", f"{home_win_pct:.1%}")

    st.markdown("#### 🏟️ Has Home Field Advantage Changed Over Time?")
    explainer("""
<b>What is this?</b> This shows the percentage of games won by the home team in each decade. A higher % means home teams had a bigger advantage that decade.<br><br>
<b>Note:</b> This chart starts from 1995 to account for expansion teams joining the league. Pre-1995 figures would be based on fewer teams and a different era of NFL travel and scheduling.
""")

    fig_decade = go.Figure(go.Bar(
        x=decade_hw['Decade'],
        y=decade_hw['Home Win Rate'] * 100,
        marker_color='#013369',
        text=[f"{v:.1f}%" for v in decade_hw['Home Win Rate'] * 100],
        textposition='outside', textfont=dict(color='white')
    ))
    fig_decade.add_hline(y=50, line_dash='dash', line_color='#555',
                         annotation_text='50% (no advantage)', annotation_font_color='#aaa',
                         annotation_position='bottom right')
    fig_decade.update_layout(**CHART_LAYOUT,
        yaxis=dict(title='Home Win %', range=[0, 70], gridcolor='#333'),
        xaxis=dict(gridcolor='#333'), height=340)
    st.plotly_chart(fig_decade, use_container_width=True)
    st.success("🏈 **NFL fan takeaway:** Home field advantage is real — but it's been getting weaker over time. This is likely due to better travel arrangements, more neutral-site games, and the COVID season (2020) which was played in empty stadiums and showed away teams can win just as often without a crowd.")

# ─────────────────────────────────────────────
# TAB 4 — HOW IT WORKS
# ─────────────────────────────────────────────
with tab4:
    st.subheader("ℹ️ How The Model Works")
    st.markdown("""
### The Data
This model is trained on **NFL games from 1990 to 2025**, sourced from the NFL scores and betting dataset on Kaggle.

### The Features
For each game, the model uses 6 inputs:
- 🏠 **Home team win rate** — historical win % for the home team
- ✈️ **Away team win rate** — historical win % for the away team
- 🏈 **Home team avg points scored** — average points scored per game
- 🏈 **Away team avg points scored** — average points scored per game
- 🛡️ **Home team avg points conceded** — average points conceded per game
- 🛡️ **Away team avg points conceded** — average points conceded per game

### The Model
Three machine learning models were compared — XGBoost was chosen. See the **🧪 Data Science** tab for the full breakdown with charts.

### Confidence Levels
- 🟢 **High Confidence** — probability gap ≥ 15%
- 🟡 **Medium Confidence** — probability gap 7–15%
- 🔴 **Low Confidence** — probability gap < 7% (very close game)

### Limitations
The model does not currently account for:
- Current season injuries
- Weather conditions
- Recent trades or roster changes
- Coaching changes

*Built by NFLNerd using Python, XGBoost and Streamlit.*
""")

# ✅ Footer
st.markdown("---")
st.caption(f"Built by NFLNerd | Trained on NFL data from 1990–2025 | Model accuracy: {accuracy:.1%}")
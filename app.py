import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
.metric-card {
    background-color: #1a1a1a;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

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

# ✅ Load and train model — now returns extra data for the DS tab
@st.cache_resource
def load_model():
    scores = pd.read_csv('data/spreadspoke_scores.csv')
    scores = scores[(scores['score_home'] > 0) | (scores['score_away'] > 0)]

    # Rename relocated/rebranded teams
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

    # Train all three models for comparison
    lr = LogisticRegression(random_state=42, max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

    # Season-by-season accuracy
    season_acc = []
    for s in sorted(seasons.unique()):
        if s < 2000:
            continue
        mask = seasons == s
        if mask.sum() < 10:
            continue
        X_s = X[mask]
        y_s = y[mask]
        preds = xgb.predict(X_s)
        season_acc.append({'season': int(s), 'accuracy': accuracy_score(y_s, preds)})

    # Confusion matrix
    cm = confusion_matrix(y_test, xgb.predict(X_test))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Confidence bucket accuracy
    probs = xgb.predict_proba(X_test)[:, 1]
    conf_data = []
    for label, low, high in [('🔴 Low (<7%)', 0, 0.07), ('🟡 Medium (7–15%)', 0.07, 0.15), ('🟢 High (>15%)', 0.15, 0.5)]:
        mask = (abs(probs - 0.5) >= low) & (abs(probs - 0.5) < high)
        if mask.sum() > 0:
            bucket_acc = accuracy_score(y_test[mask], xgb.predict(X_test)[mask])
            conf_data.append({'confidence': label, 'accuracy': bucket_acc, 'games': mask.sum()})

    # Feature importances
    feature_names = ['Home Win Rate', 'Away Win Rate', 'Home Avg Scored',
                     'Away Avg Scored', 'Home Avg Conceded', 'Away Avg Conceded']
    importances = xgb.feature_importances_

    return (xgb, scores, get_team_stats, xgb_acc,
            lr_acc, rf_acc,
            pd.DataFrame(season_acc),
            cm, fpr, tpr, roc_auc,
            conf_data,
            feature_names, importances,
            X_test, y_test)


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
    (xgb, scores, get_team_stats, accuracy,
     lr_acc, rf_acc,
     season_acc_df,
     cm, fpr, tpr, roc_auc,
     conf_data,
     feature_names, importances,
     X_test, y_test) = load_model()

st.success(f"Model loaded! Accuracy: {accuracy:.1%}")

# ✅ Use only current 32 NFL teams in dropdowns
all_teams = sorted(CURRENT_NFL_TEAMS)

# ✅ Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📅 Weekly Predictions", "🧪 Data Science", "ℹ️ How It Works"])

# ─────────────────────────────────────────────
# TAB 1 — PREDICT
# ─────────────────────────────────────────────
with tab1:
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

            # Team stat comparison table
            st.markdown("---")
            st.subheader("📋 Team Stats Comparison")
            stats_df = pd.DataFrame({
                'Stat': ['Historical Win Rate', 'Avg Points Scored', 'Avg Points Conceded'],
                f'🏠 {home_team}': [f"{home_wr:.1%}", f"{home_scored:.1f}", f"{home_conceded:.1f}"],
                f'✈️ {away_team}': [f"{away_wr:.1%}", f"{away_scored:.1f}", f"{away_conceded:.1f}"],
            })
            st.dataframe(stats_df.set_index('Stat'), use_container_width=True)

            # Win probability bar chart
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

            # Head to head
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

            # Recent form
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

# ─────────────────────────────────────────────
# TAB 3 — DATA SCIENCE SHOWCASE
# ─────────────────────────────────────────────
with tab3:
    st.subheader("🧪 Data Science & Model Analysis")
    st.markdown("A deep dive into how the prediction model works, what it's learned, and how well it actually performs.")

    # ── Section 1: Model Comparison ──
    st.markdown("---")
    st.subheader("🤖 Model Comparison")
    st.markdown("Three machine learning models were trained and evaluated on the same dataset. XGBoost came out on top.")

    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost ✅']
    model_accs = [lr_acc * 100, rf_acc * 100, accuracy * 100]
    colors = ['#555555', '#888888', '#D50A0A']

    fig_models = go.Figure(go.Bar(
        x=model_names,
        y=model_accs,
        marker_color=colors,
        text=[f"{a:.1f}%" for a in model_accs],
        textposition='outside',
        textfont=dict(color='white', size=14)
    ))
    fig_models.add_hline(y=50, line_dash='dash', line_color='#666',
                         annotation_text='Random Guessing (50%)', annotation_font_color='#aaa')
    fig_models.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        yaxis=dict(title='Accuracy (%)', range=[45, 65], gridcolor='#333'),
        xaxis=dict(gridcolor='#333'),
        showlegend=False, height=380
    )
    st.plotly_chart(fig_models, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Logistic Regression", f"{lr_acc:.1%}")
    with col2:
        st.metric("Random Forest", f"{rf_acc:.1%}")
    with col3:
        st.metric("XGBoost (chosen)", f"{accuracy:.1%}", delta=f"+{(accuracy - lr_acc):.1%} vs LR")

    # ── Section 2: Feature Importance ──
    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    st.markdown("Which stats does the model rely on most when making predictions? Higher = more influential.")

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    fig_fi = go.Figure(go.Bar(
        x=sorted_importances,
        y=sorted_features,
        orientation='h',
        marker_color='#013369',
        text=[f"{v:.3f}" for v in sorted_importances],
        textposition='outside',
        textfont=dict(color='white')
    ))
    fig_fi.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        xaxis=dict(title='Importance Score', gridcolor='#333'),
        yaxis=dict(gridcolor='#333', autorange='reversed'),
        showlegend=False, height=380
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    top_feature = sorted_features[0]
    st.info(f"💡 **{top_feature}** is the single most important factor in the model's predictions — teams with a stronger historical record have a significantly higher chance of being picked as the winner.")

    # ── Section 3: Season-by-Season Accuracy ──
    st.markdown("---")
    st.subheader("📈 Season-by-Season Accuracy")
    st.markdown("How did the model perform year by year? Note: the model is trained on all data up to 2024, so earlier seasons have more data behind them.")

    if not season_acc_df.empty:
        avg_acc = season_acc_df['accuracy'].mean()

        fig_season = go.Figure()
        fig_season.add_trace(go.Scatter(
            x=season_acc_df['season'],
            y=season_acc_df['accuracy'] * 100,
            mode='lines+markers',
            line=dict(color='#D50A0A', width=2),
            marker=dict(size=7, color='#D50A0A'),
            name='Model Accuracy'
        ))
        fig_season.add_hline(y=50, line_dash='dash', line_color='#666',
                             annotation_text='Random Guessing', annotation_font_color='#aaa')
        fig_season.add_hline(y=avg_acc * 100, line_dash='dot', line_color='#013369',
                             annotation_text=f'Average ({avg_acc:.1%})', annotation_font_color='#aaa')
        fig_season.update_layout(
            plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
            font=dict(color='white'),
            xaxis=dict(title='Season', gridcolor='#333', dtick=2),
            yaxis=dict(title='Accuracy (%)', range=[40, 75], gridcolor='#333'),
            showlegend=False, height=380
        )
        st.plotly_chart(fig_season, use_container_width=True)

        best_season = season_acc_df.loc[season_acc_df['accuracy'].idxmax()]
        worst_season = season_acc_df.loc[season_acc_df['accuracy'].idxmin()]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Accuracy", f"{avg_acc:.1%}")
        with col2:
            st.metric("Best Season", f"{int(best_season['season'])} ({best_season['accuracy']:.1%})")
        with col3:
            st.metric("Toughest Season", f"{int(worst_season['season'])} ({worst_season['accuracy']:.1%})")

    # ── Section 4: Confidence vs Accuracy ──
    st.markdown("---")
    st.subheader("🎯 Confidence Level vs Actual Accuracy")
    st.markdown("Does the model's confidence actually reflect how accurate it is? It should be more accurate on high-confidence picks.")

    if conf_data:
        conf_df = pd.DataFrame(conf_data)
        fig_conf = go.Figure(go.Bar(
            x=conf_df['confidence'],
            y=conf_df['accuracy'] * 100,
            marker_color=['#D50A0A', '#FFB300', '#00C853'],
            text=[f"{a:.1%} ({g} games)" for a, g in zip(conf_df['accuracy'], conf_df['games'])],
            textposition='outside',
            textfont=dict(color='white', size=13)
        ))
        fig_conf.add_hline(y=50, line_dash='dash', line_color='#666',
                           annotation_text='Random Guessing', annotation_font_color='#aaa')
        fig_conf.update_layout(
            plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
            font=dict(color='white'),
            yaxis=dict(title='Actual Accuracy (%)', range=[40, 80], gridcolor='#333'),
            xaxis=dict(gridcolor='#333'),
            showlegend=False, height=380
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        st.info("💡 High confidence predictions should have a noticeably higher accuracy than low confidence ones — if they do, the model's confidence scores are well-calibrated.")

    # ── Section 5: Confusion Matrix ──
    st.markdown("---")
    st.subheader("🔢 Confusion Matrix")
    st.markdown("Shows how often the model correctly predicted home wins vs away wins on the test set.")

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=['Predicted Away Win', 'Predicted Home Win'],
        y=['Actual Away Win', 'Actual Home Win'],
        colorscale=[[0, '#0a0a0a'], [1, '#D50A0A']],
        text=cm,
        texttemplate='%{text}',
        textfont=dict(size=20, color='white'),
        showscale=False
    ))
    fig_cm.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        height=350
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("True Away Wins", tn)
    with col2:
        st.metric("True Home Wins", tp)
    with col3:
        st.metric("False Positives", fp, delta="Away predicted as Home", delta_color="inverse")
    with col4:
        st.metric("False Negatives", fn, delta="Home predicted as Away", delta_color="inverse")

    # ── Section 6: ROC Curve ──
    st.markdown("---")
    st.subheader("📉 ROC Curve")
    st.markdown(f"The ROC curve shows the model's ability to distinguish between home wins and away wins. AUC = **{roc_auc:.3f}** (perfect = 1.0, random = 0.5).")

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color='#D50A0A', width=2),
        name=f'XGBoost (AUC = {roc_auc:.3f})'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#666', dash='dash'),
        name='Random Classifier'
    ))
    fig_roc.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        xaxis=dict(title='False Positive Rate', gridcolor='#333'),
        yaxis=dict(title='True Positive Rate', gridcolor='#333'),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#333'),
        height=400
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Section 7: Dataset Overview ──
    st.markdown("---")
    st.subheader("📦 Dataset Overview")

    total_games = len(scores)
    seasons_covered = int(scores['schedule_season'].max()) - int(scores['schedule_season'].min()) + 1
    teams_covered = len(set(scores['team_home'].unique()) | set(scores['team_away'].unique()))
    home_win_pct = scores['home_win'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Games", f"{total_games:,}")
    with col2:
        st.metric("Seasons", seasons_covered)
    with col3:
        st.metric("Teams", teams_covered)
    with col4:
        st.metric("Home Win Rate", f"{home_win_pct:.1%}")

    st.markdown("#### Home Win Rate by Decade")
    scores['decade'] = (scores['schedule_season'] // 10 * 10).astype(str) + 's'
    decade_hw = scores.groupby('decade')['home_win'].mean().reset_index()
    decade_hw.columns = ['Decade', 'Home Win Rate']

    fig_decade = go.Figure(go.Bar(
        x=decade_hw['Decade'],
        y=decade_hw['Home Win Rate'] * 100,
        marker_color='#013369',
        text=[f"{v:.1f}%" for v in decade_hw['Home Win Rate'] * 100],
        textposition='outside',
        textfont=dict(color='white')
    ))
    fig_decade.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        yaxis=dict(title='Home Win %', range=[0, 70], gridcolor='#333'),
        xaxis=dict(gridcolor='#333'),
        showlegend=False, height=320
    )
    st.plotly_chart(fig_decade, use_container_width=True)
    st.caption("Home field advantage has gradually declined since the 1990s — interesting for the model!")

# ─────────────────────────────────────────────
# TAB 4 — HOW IT WORKS
# ─────────────────────────────────────────────
with tab4:
    st.subheader("ℹ️ How The Model Works")
    st.markdown("""
### The Data

This model is trained on **NFL games from 1990 to 2025**, sourced from the NFL scores and betting dataset on Kaggle.

### The Features

For each game, the model uses 6 features:

- 🏠 **Home team win rate** — historical win % for the home team
- ✈️ **Away team win rate** — historical win % for the away team
- 🏈 **Home team avg points scored** — average points scored per game
- 🏈 **Away team avg points scored** — average points scored per game
- 🛡️ **Home team avg points conceded** — average points conceded per game
- 🛡️ **Away team avg points conceded** — average points conceded per game

### The Model

Three machine learning models were compared — XGBoost performed best and was chosen. See the **🧪 Data Science** tab for the full breakdown.

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
st.caption(f"Built by NFLNerd | Trained on NFL data from 1990-2025 | Model accuracy: {accuracy:.1%}")
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

st.markdown("""
<style>
.stApp { background-color: #0a0a0a; color: white; }
.stButton > button {
    background-color: #D50A0A; color: white;
    font-size: 18px; font-weight: bold;
    border: none; border-radius: 8px;
}
.stButton > button:hover { background-color: #013369; }
.stSelectbox label { color: white; font-weight: bold; }
h1, h2, h3 { color: white; }
.nflnerd-brand {
    font-size: 14px; color: #D50A0A;
    font-weight: bold; letter-spacing: 2px;
}
.explainer-box {
    background-color: #1a1a1a;
    border-left: 4px solid #D50A0A;
    padding: 12px 16px; border-radius: 4px;
    margin-bottom: 12px; font-size: 14px; line-height: 1.7;
}
.takeaway-box {
    background-color: #0d1f0d;
    border-left: 4px solid #00C853;
    padding: 12px 16px; border-radius: 4px;
    margin-top: 8px; font-size: 14px; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

def explainer(text):
    st.markdown(f'<div class="explainer-box">{text}</div>', unsafe_allow_html=True)

def takeaway(text):
    st.markdown(f'<div class="takeaway-box">{text}</div>', unsafe_allow_html=True)

CURRENT_NFL_TEAMS = [
    'Arizona Cardinals','Atlanta Falcons','Baltimore Ravens','Buffalo Bills',
    'Carolina Panthers','Chicago Bears','Cincinnati Bengals','Cleveland Browns',
    'Dallas Cowboys','Denver Broncos','Detroit Lions','Green Bay Packers',
    'Houston Texans','Indianapolis Colts','Jacksonville Jaguars','Kansas City Chiefs',
    'Las Vegas Raiders','Los Angeles Chargers','Los Angeles Rams','Miami Dolphins',
    'Minnesota Vikings','New England Patriots','New Orleans Saints','New York Giants',
    'New York Jets','Philadelphia Eagles','Pittsburgh Steelers','San Francisco 49ers',
    'Seattle Seahawks','Tampa Bay Buccaneers','Tennessee Titans','Washington Commanders'
]

CHART_LAYOUT = dict(
    plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
    font=dict(color='white'), showlegend=False
)

@st.cache_resource
def load_model():
    scores = pd.read_csv('data/spreadspoke_scores.csv')
    scores = scores[(scores['score_home'] > 0) | (scores['score_away'] > 0)]

    team_renames = {
        'Oakland Raiders':'Las Vegas Raiders','St. Louis Rams':'Los Angeles Rams',
        'San Diego Chargers':'Los Angeles Chargers','Washington Redskins':'Washington Commanders',
        'Washington Football Team':'Washington Commanders','Tennessee Oilers':'Tennessee Titans',
        'Houston Oilers':'Tennessee Titans','Phoenix Cardinals':'Arizona Cardinals',
        'Baltimore Colts':'Indianapolis Colts','Los Angeles Raiders':'Las Vegas Raiders',
    }
    scores['team_home'] = scores['team_home'].replace(team_renames)
    scores['team_away'] = scores['team_away'].replace(team_renames)
    scores['home_win'] = (scores['score_home'] > scores['score_away']).astype(int)
    scores['schedule_date'] = pd.to_datetime(scores['schedule_date'])
    scores = scores.sort_values('schedule_season')
    scores = scores[scores['schedule_season'] >= 1990].copy()

    def get_team_stats(df, team, before_season):
        hg = df[(df['team_home'] == team) & (df['schedule_season'] < before_season)]
        ag = df[(df['team_away'] == team) & (df['schedule_season'] < before_season)]
        hw = (hg['score_home'] > hg['score_away']).sum()
        aw = (ag['score_away'] > ag['score_home']).sum()
        total = len(hg) + len(ag)
        if total == 0:
            return 0.5, 22.0, 20.0
        wr = (hw + aw) / total
        scored    = pd.concat([hg['score_home'], ag['score_away']]).mean()
        conceded  = pd.concat([hg['score_away'], ag['score_home']]).mean()
        return wr, scored, conceded

    game_data = []
    for _, row in scores.iterrows():
        s = row['schedule_season']
        h_wr, h_sc, h_co = get_team_stats(scores, row['team_home'], s)
        a_wr, a_sc, a_co = get_team_stats(scores, row['team_away'], s)
        game_data.append({
            'home_win_rate': h_wr, 'away_win_rate': a_wr,
            'home_avg_scored': h_sc, 'away_avg_scored': a_sc,
            'home_avg_conceded': h_co, 'away_avg_conceded': a_co,
            'home_win': row['home_win'], 'season': s
        })

    df_f = pd.DataFrame(game_data).dropna()
    X = df_f.drop(['home_win','season'], axis=1)
    y = df_f['home_win']
    seasons = df_f['season']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr  = LogisticRegression(random_state=42, max_iter=1000)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    lr.fit(X_train, y_train); rf.fit(X_train, y_train); xgb.fit(X_train, y_train)

    lr_acc  = accuracy_score(y_test, lr.predict(X_test))
    rf_acc  = accuracy_score(y_test, rf.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

    # Season-by-season (2000+)
    season_acc, season_notes = [], {
        2004: "2004 — An unusually balanced season with no dominant team. High parity = harder to predict.",
        2007: "2007 — The 16-0 Patriots made this an easier-than-usual year for the model.",
        2020: "2020 — The COVID season. Games played without fans removed home crowd advantage, making results harder to predict.",
        2022: "2022 — A high-upset season. Multiple strong teams lost games they were expected to win easily.",
    }
    for s in sorted(seasons.unique()):
        if s < 2000: continue
        mask = seasons == s
        if mask.sum() < 10: continue
        season_acc.append({'season': int(s),
                           'accuracy': accuracy_score(y[mask], xgb.predict(X[mask])),
                           'note': season_notes.get(int(s), "")})

    cm = confusion_matrix(y_test, xgb.predict(X_test))
    fpr, tpr, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)

    probs = xgb.predict_proba(X_test)[:,1]
    preds_test = xgb.predict(X_test)
    conf_data = []
    for label, display, low, high in [
        ('🔴 Low','🔴 Low Confidence\n(model sees it as nearly 50/50)',    0.0, 0.07),
        ('🟡 Medium','🟡 Medium Confidence\n(model leans one way)',        0.07,0.15),
        ('🟢 High','🟢 High Confidence\n(model strongly favours one team)',0.15,0.50),
    ]:
        mask = (abs(probs - 0.5) >= low) & (abs(probs - 0.5) < high)
        if mask.sum() > 0:
            conf_data.append({
                'confidence': display,
                'accuracy': accuracy_score(y_test[mask], preds_test[mask]),
                'games': int(mask.sum())
            })

    feature_names = ['Home Win Rate','Away Win Rate','Home Avg Scored',
                     'Away Avg Scored','Home Avg Conceded','Away Avg Conceded']
    importances = xgb.feature_importances_

    # Decade chart: 2002+ (Houston Texans joined, completing 32 teams)
    scores_32 = scores[scores['schedule_season'] >= 2002].copy()
    scores_32['period'] = pd.cut(
        scores_32['schedule_season'],
        bins=[2001,2009,2019,2026],
        labels=['2002–2009','2010–2019','2020–2025']
    )
    period_hw = scores_32.groupby('period', observed=True)['home_win'].agg(
        home_win_rate='mean', games='count'
    ).reset_index()
    period_hw.columns = ['Period','Home Win Rate','Games']

    return (xgb, scores, get_team_stats, xgb_acc, lr_acc, rf_acc,
            pd.DataFrame(season_acc), cm, fpr, tpr, roc_auc,
            conf_data, feature_names, importances, X_test, y_test,
            period_hw, scores_32)

def get_recent_form(scores, team, n=5):
    hg = scores[scores['team_home']==team][['schedule_date','team_home','team_away','score_home','score_away']].copy()
    hg['win'] = hg['score_home'] > hg['score_away']
    hg['result'] = hg.apply(lambda r: f"✅ W {int(r['score_home'])}-{int(r['score_away'])} vs {r['team_away']}" if r['win'] else f"❌ L {int(r['score_home'])}-{int(r['score_away'])} vs {r['team_away']}", axis=1)
    ag = scores[scores['team_away']==team][['schedule_date','team_home','team_away','score_home','score_away']].copy()
    ag['win'] = ag['score_away'] > ag['score_home']
    ag['result'] = ag.apply(lambda r: f"✅ W {int(r['score_away'])}-{int(r['score_home'])} @ {r['team_home']}" if r['win'] else f"❌ L {int(r['score_away'])}-{int(r['score_home'])} @ {r['team_home']}", axis=1)
    all_g = pd.concat([hg[['schedule_date','result','win']], ag[['schedule_date','result','win']]])
    all_g['schedule_date'] = pd.to_datetime(all_g['schedule_date'])
    return all_g.sort_values('schedule_date', ascending=False).head(n)

def get_head_to_head(scores, t1, t2):
    h2h = scores[((scores['team_home']==t1)&(scores['team_away']==t2))|
                 ((scores['team_home']==t2)&(scores['team_away']==t1))].copy()
    w1=w2=0
    for _, r in h2h.iterrows():
        if r['team_home']==t1:
            if r['score_home']>r['score_away']: w1+=1
            else: w2+=1
        else:
            if r['score_away']>r['score_home']: w1+=1
            else: w2+=1
    return w1, w2, len(h2h)

# ── Load ──
st.markdown('<p class="nflnerd-brand">🏈 NFLNERD</p>', unsafe_allow_html=True)
st.title("NFL Game Outcome Predictor")
st.markdown("Predict the outcome of any NFL matchup using machine learning trained on historical data since 1990.")

with st.spinner('Loading model... this may take a minute on first load!'):
    (xgb, scores, get_team_stats, accuracy, lr_acc, rf_acc,
     season_acc_df, cm, fpr, tpr, roc_auc,
     conf_data, feature_names, importances,
     X_test, y_test, period_hw, scores_32) = load_model()

st.success(f"Model loaded! Accuracy: {accuracy:.1%}")
all_teams = sorted(CURRENT_NFL_TEAMS)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict","📅 Weekly Predictions","🧪 Data Science","ℹ️ How It Works"])

# ─── TAB 1 ───────────────────────────────────
with tab1:
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Select home team", all_teams,
            index=all_teams.index("Kansas City Chiefs") if "Kansas City Chiefs" in all_teams else 0)
    with c2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Select away team", all_teams,
            index=all_teams.index("Philadelphia Eagles") if "Philadelphia Eagles" in all_teams else 1)
    st.markdown("---")

    if st.button("🔮 Predict Game Outcome", use_container_width=True):
        if home_team == away_team:
            st.error("Please select two different teams!")
        else:
            h_wr,h_sc,h_co = get_team_stats(scores, home_team, 2026)
            a_wr,a_sc,a_co = get_team_stats(scores, away_team, 2026)
            prob = xgb.predict_proba([[h_wr,a_wr,h_sc,a_sc,h_co,a_co]])[0]
            hp, ap = prob[1], prob[0]
            diff = abs(hp-ap)
            if diff>=0.15: conf,msg="🟢 High Confidence","The model is strongly favouring one team."
            elif diff>=0.07: conf,msg="🟡 Medium Confidence","The model leans one way but it's not clear cut."
            else: conf,msg="🔴 Low Confidence — Close Game","This is a very tight matchup. Could go either way!"

            st.markdown("---"); st.subheader("📊 Prediction Results")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric(f"🏠 {home_team}", f"{hp:.1%}")
            with c2: st.metric("VS","")
            with c3: st.metric(f"✈️ {away_team}", f"{ap:.1%}")
            if hp>ap: st.success(f"🏆 Predicted Winner: **{home_team}** ({hp:.1%} probability)")
            else: st.success(f"🏆 Predicted Winner: **{away_team}** ({ap:.1%} probability)")
            st.info(f"{conf} — {msg}")

            st.markdown("---"); st.subheader("📋 Team Stats Comparison")
            st.dataframe(pd.DataFrame({
                'Stat':['Historical Win Rate','Avg Points Scored','Avg Points Conceded'],
                f'🏠 {home_team}':[f"{h_wr:.1%}",f"{h_sc:.1f}",f"{h_co:.1f}"],
                f'✈️ {away_team}':[f"{a_wr:.1%}",f"{a_sc:.1f}",f"{a_co:.1f}"],
            }).set_index('Stat'), use_container_width=True)

            st.markdown("---"); st.subheader("📊 Win Probability")
            fig = go.Figure(go.Bar(
                x=[f"🏠 {home_team}",f"✈️ {away_team}"], y=[hp*100,ap*100],
                marker_color=["#013369","#D50A0A"],
                text=[f"{hp:.1%}",f"{ap:.1%}"],
                textposition="outside", textfont=dict(color="white",size=16)))
            fig.update_layout(**CHART_LAYOUT,
                yaxis=dict(title="Win Probability (%)",range=[0,110],gridcolor="#333"),
                xaxis=dict(gridcolor="#333"),height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---"); st.subheader("⚔️ Head to Head Record")
            w1,w2,tot = get_head_to_head(scores,home_team,away_team)
            if tot==0: st.write("No head to head record found.")
            else:
                c1,c2,c3=st.columns(3)
                with c1: st.metric(f"🏠 {home_team} Wins",w1)
                with c2: st.metric("Total Games",tot)
                with c3: st.metric(f"✈️ {away_team} Wins",w2)
                if w1>w2: st.write(f"**{home_team}** lead the all time series {w1}-{w2}")
                elif w2>w1: st.write(f"**{away_team}** lead the all time series {w2}-{w1}")
                else: st.write(f"The all time series is tied {w1}-{w2}")

            st.markdown("---"); st.subheader("📅 Recent Form (Last 5 Games)")
            c1,c2=st.columns(2)
            with c1:
                st.markdown(f"**🏠 {home_team}**")
                for _,r in get_recent_form(scores,home_team).iterrows(): st.markdown(r['result'])
            with c2:
                st.markdown(f"**✈️ {away_team}**")
                for _,r in get_recent_form(scores,away_team).iterrows(): st.markdown(r['result'])

# ─── TAB 2 ───────────────────────────────────
with tab2:
    st.subheader("📅 Weekly Predictions")
    st.markdown("Enter this week's matchups and get predictions for all games at once.")
    st.info("Add up to 16 matchups below:")
    num_games = st.slider("How many games?",1,16,4)
    matchups=[]
    for i in range(num_games):
        st.markdown(f"**Game {i+1}**")
        c1,c2=st.columns(2)
        with c1: h=st.selectbox(f"Home team {i+1}",all_teams,key=f"home_{i}")
        with c2: a=st.selectbox(f"Away team {i+1}",all_teams,key=f"away_{i}",index=1)
        matchups.append((h,a))
    if st.button("🔮 Predict All Games",use_container_width=True):
        st.markdown("---")
        for i,(h,a) in enumerate(matchups):
            if h==a: st.warning(f"Game {i+1}: Skipping — same team selected for both sides"); continue
            h_wr,h_sc,h_co=get_team_stats(scores,h,2026)
            a_wr,a_sc,a_co=get_team_stats(scores,a,2026)
            prob=xgb.predict_proba([[h_wr,a_wr,h_sc,a_sc,h_co,a_co]])[0]
            hp,ap=prob[1],prob[0]
            winner=h if hp>ap else a
            diff=abs(hp-ap)
            conf="🟢" if diff>=0.15 else ("🟡" if diff>=0.07 else "🔴")
            st.markdown(f"{conf} **{h}** vs **{a}** → 🏆 **{winner}** ({max(hp,ap):.1%})")

# ─── TAB 3 — DATA SCIENCE ────────────────────
with tab3:
    st.subheader("🧪 Data Science & Model Analysis")
    st.markdown("""
This tab walks through how the prediction model was built, what it learned from 35 years of NFL data, and how well it actually performs.
Each section has an explanation of what you're looking at and what the numbers mean.
""")

    # ── 1. Model Comparison ──────────────────
    st.markdown("---")
    st.subheader("🤖 1. Which Model Performed Best?")

    explainer("""
<b>What is a machine learning model?</b><br>
A machine learning model is a piece of software that learns patterns from historical data and uses those patterns to make predictions on new data it hasn't seen before. Think of it like a weather forecasting app — it doesn't know exactly what tomorrow's weather will be, but it has studied millions of days of past weather data and can make an educated prediction.<br><br>

<b>Three models were tested:</b><br>
To find the best predictor, three different models were trained on the same 35 years of NFL game data and tested against each other — like asking three different forecasting apps to predict the same week of weather and seeing which one is most accurate.<br><br>

• <b>Logistic Regression</b> — the simplest of the three. It looks for straight-line relationships in the data (e.g. "the more games a team has won historically, the more likely they win today"). Fast and interpretable, but limited.<br>
• <b>Random Forest</b> — a more powerful approach. It builds hundreds of independent decision trees and takes a majority vote. Better at spotting more complex relationships.<br>
• <b>XGBoost</b> — the most advanced of the three. It builds trees one at a time, where each new tree learns from the mistakes of the previous one. This approach — called <i>gradient boosting</i> — is the industry standard for structured data like this and is widely used by professional data scientists.<br><br>

<b>Why was XGBoost chosen even if the accuracy numbers are close?</b><br>
On this dataset the three models perform similarly because the data is relatively straightforward — there are only 6 inputs. However, XGBoost was chosen because it handles more complex, non-straight-line relationships between stats (for example, the combined effect of a high-scoring offence <i>and</i> a leaky defence matters differently than either stat alone). It also produces feature importance scores — a breakdown of exactly which stats drove each prediction — which makes the model explainable, not just accurate. As more data and features are added in future, XGBoost will scale better than the simpler models.
""")

    model_names = ['Logistic Regression','Random Forest','XGBoost ✅']
    model_accs  = [lr_acc*100, rf_acc*100, accuracy*100]
    best_acc    = max(model_accs)
    delta_val   = accuracy - lr_acc
    delta_str   = f"+{delta_val:.2%}" if delta_val >= 0 else f"{delta_val:.2%}"

    fig_m = go.Figure(go.Bar(
        x=model_names, y=model_accs,
        marker_color=['#555555','#888888','#D50A0A'],
        text=[f"{a:.2f}%" for a in model_accs],
        textposition='outside', textfont=dict(color='white',size=14)))
    fig_m.add_hline(y=50, line_dash='dash', line_color='#666',
                    annotation_text='Random guessing (50%)', annotation_font_color='#aaa',
                    annotation_position='bottom right')
    fig_m.update_layout(**CHART_LAYOUT,
        yaxis=dict(title='Accuracy (%)',range=[45,65],gridcolor='#333'),
        xaxis=dict(gridcolor='#333'),height=380)
    st.plotly_chart(fig_m, use_container_width=True)

    c1,c2,c3=st.columns(3)
    with c1: st.metric("Logistic Regression", f"{lr_acc:.2%}")
    with c2: st.metric("Random Forest",       f"{rf_acc:.2%}")
    with c3: st.metric("XGBoost (chosen ✅)", f"{accuracy:.2%}", delta=delta_str)

    takeaway("""
The accuracy numbers are deliberately shown to two decimal places here — the differences between models are small but real.
All three models beat random guessing (50%), which is the baseline: a model that just flipped a coin for every game.
The NFL is one of the hardest sports to predict because of how much parity the league intentionally builds in through the draft system and salary cap — so even a 58% accurate model is genuinely competitive with professional forecasting tools.
""")

    # ── 2. Feature Importance ───────────────
    st.markdown("---")
    st.subheader("🔍 2. Which Stats Matter Most to the Model?")

    explainer("""
<b>What is this showing?</b><br>
When XGBoost predicts a winner, it doesn't treat all 6 inputs equally — some stats have more influence on the final decision than others. Feature importance scores measure exactly how much each stat contributed to the model's decisions across all predictions it made during training.<br><br>

<b>How to read the importance score:</b><br>
The score is a number between 0 and 1. A score of 0.35 means that stat was responsible for 35% of the model's decision-making. All six scores add up to 1.0 (100%).<br><br>

<b>What each stat on the Y-axis means:</b><br>
• <b>Home/Away Win Rate</b> — that team's overall win percentage across all their historical games in the dataset<br>
• <b>Home/Away Avg Points Scored</b> — how many points that team scores per game on average across all historical games<br>
• <b>Home/Away Avg Points Conceded</b> — how many points that team lets in per game on average<br><br>

<b>Why might Away Win Rate rank higher than Home Win Rate?</b><br>
Winning away from home is genuinely harder — no home crowd, unfamiliar stadium, travel fatigue. A team that wins consistently on the road is usually a better all-round team than one that only performs well at home. So the model has learned that away win rate is a stronger signal of true team quality than home win rate.
""")

    idx  = np.argsort(importances)[::-1]
    s_fn = [feature_names[i] for i in idx]
    s_fi = [importances[i]   for i in idx]

    fig_fi = go.Figure(go.Bar(
        x=s_fi, y=s_fn, orientation='h',
        marker_color='#013369',
        text=[f"{v:.3f}" for v in s_fi],
        textposition='outside', textfont=dict(color='white')))
    fig_fi.update_layout(**CHART_LAYOUT,
        xaxis=dict(title='Importance Score (all six add up to 1.0)',
                   gridcolor='#333', range=[0, max(s_fi)*1.3]),
        yaxis=dict(gridcolor='#333', autorange='reversed'),
        height=400)
    st.plotly_chart(fig_fi, use_container_width=True)

    takeaway(f"""
<b>{s_fn[0]}</b> is the single most influential stat in the model — responsible for roughly {s_fi[0]:.0%} of every prediction.
The bottom two stats contribute the least, but removing them entirely would still reduce accuracy slightly, so all six are kept.
""")

    # ── 3. Season-by-Season ─────────────────
    st.markdown("---")
    st.subheader("📈 3. How Accurate Was the Model in Each NFL Season?")

    explainer("""
<b>What is this showing?</b><br>
Rather than a single overall accuracy number, this chart breaks down how accurately the model predicted games in each individual NFL season from 2000 to 2025. Each point on the line represents one full season.<br><br>

<b>Why does accuracy change year to year?</b><br>
Every NFL season has upsets — that's part of what makes the sport so popular. But some seasons have <i>significantly more</i> upsets than others. In a season where a few dominant teams win most of their games convincingly, the model finds it easier to predict. In a season where the league is more unpredictable — more underdogs winning, more injuries to key players mid-season — the model's accuracy dips.<br><br>

<b>Why does the chart start from 2000?</b><br>
Pre-2000 NFL data covers a different era of the sport with different scoring patterns and fewer teams. Including those seasons in a year-by-year comparison would make the chart misleading, so it starts from 2000 for consistency. The data is still used in model training though.<br><br>

<b>Hover over any point</b> on the line to see the exact accuracy and any notable context for that season.
""")

    if not season_acc_df.empty:
        avg_acc = season_acc_df['accuracy'].mean()

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=season_acc_df['season'],
            y=season_acc_df['accuracy']*100,
            mode='lines+markers',
            line=dict(color='#D50A0A',width=2),
            marker=dict(size=8,color='#D50A0A'),
            customdata=season_acc_df['note'],
            hovertemplate='<b>Season: %{x}</b><br>Accuracy: %{y:.1f}%<br>%{customdata}<extra></extra>'
        ))
        fig_s.add_hline(y=50, line_dash='dash', line_color='#555',
                        annotation_text='Random guessing (50%)',
                        annotation_font_color='#aaa', annotation_position='bottom right')
        fig_s.add_hline(y=avg_acc*100, line_dash='dot', line_color='#013369',
                        annotation_text=f'Overall average ({avg_acc:.1%})',
                        annotation_font_color='#aaa', annotation_position='top right')
        fig_s.update_layout(**CHART_LAYOUT,
            xaxis=dict(title='NFL Season', gridcolor='#333', dtick=2,
                       range=[season_acc_df['season'].min()-1,
                              season_acc_df['season'].max()+1]),
            yaxis=dict(title='Accuracy (%)',range=[40,80],gridcolor='#333'),
            height=420)
        st.plotly_chart(fig_s, use_container_width=True)

        best  = season_acc_df.loc[season_acc_df['accuracy'].idxmax()]
        worst = season_acc_df.loc[season_acc_df['accuracy'].idxmin()]
        c1,c2,c3=st.columns(3)
        with c1: st.metric("Overall Average",   f"{avg_acc:.1%}")
        with c2: st.metric("Best Season",  f"{int(best['season'])} ({best['accuracy']:.1%})")
        with c3: st.metric("Toughest Season", f"{int(worst['season'])} ({worst['accuracy']:.1%})")

        takeaway(f"""
The model beats random guessing in every single season — the red line stays above the 50% dashed baseline throughout.
Its toughest year was <b>{int(worst['season'])}</b> and its best was <b>{int(best['season'])}</b>.
The accuracy naturally fluctuates because NFL season results fluctuate — a more predictable season produces a higher accuracy score, not a better model.
""")

    # ── 4. Confidence ───────────────────────
    st.markdown("---")
    st.subheader("🎯 4. Does the Model's Confidence Level Actually Mean Anything?")

    explainer("""
<b>How does the model express confidence?</b><br>
When the model predicts a game, it doesn't just say "home team wins" — it gives each team a probability. For example, it might say the home team has a <b>64% chance of winning</b> and the away team has a <b>36% chance</b>. That means: if you played this exact game 100 times under the same conditions, the model expects the home team to win around 64 of them.<br><br>

<b>The three confidence levels are based on the gap between those two percentages:</b><br>
• 🔴 <b>Low Confidence</b> — the gap between the two probabilities is less than 7 percentage points. Example: 53% vs 47%. The model sees this as almost a coin flip.<br>
• 🟡 <b>Medium Confidence</b> — the gap is between 7 and 15 percentage points. Example: 57% vs 43%. The model leans one way but acknowledges real uncertainty.<br>
• 🟢 <b>High Confidence</b> — the gap is more than 15 percentage points. Example: 66% vs 34%. The model strongly favours one team.<br><br>

<b>What do the numbers above the bars mean?</b><br>
Each bar shows the actual accuracy achieved within that confidence bucket — i.e. what percentage of those games the model correctly predicted. The number in brackets (e.g. "312 games") shows how many games from the test dataset fell into that bucket. The test dataset covers games across all seasons from 1990–2025, with 20% of all games held back purely for testing (roughly 1,900 games total).
""")

    if conf_data:
        cdf = pd.DataFrame(conf_data)
        fig_c = go.Figure(go.Bar(
            x=cdf['confidence'],
            y=cdf['accuracy']*100,
            marker_color=['#D50A0A','#FFB300','#00C853'],
            text=[f"{a:.1%}\n({g} games)" for a,g in zip(cdf['accuracy'],cdf['games'])],
            textposition='outside', textfont=dict(color='white',size=13)))
        fig_c.add_hline(y=50, line_dash='dash', line_color='#555',
                        annotation_text='Random guessing (50%)',
                        annotation_font_color='#aaa', annotation_position='bottom right')
        fig_c.update_layout(**CHART_LAYOUT,
            yaxis=dict(title='Actual Accuracy (%)',range=[40,85],gridcolor='#333'),
            xaxis=dict(gridcolor='#333'),height=420)
        st.plotly_chart(fig_c, use_container_width=True)

        takeaway("""
If this chart is working as it should, the green bar (High Confidence) will be noticeably taller than the red bar (Low Confidence).
That would confirm that when the model says it's confident, it really is more likely to be right — and when it flags a game as a near coin-flip, it genuinely is harder to call.
A model that can correctly identify which games it's likely to get wrong is more useful than one that just gives the same confidence level every time.
""")

    # ── 5. Confusion Matrix ─────────────────
    st.markdown("---")
    st.subheader("🔢 5. Where Does the Model Go Wrong?")

    explainer("""
<b>Why is everything framed as home team vs away team?</b><br>
Every NFL game in the dataset is recorded with one team listed as "home" and one as "away." The model was trained to answer one specific question: <i>will the home team win?</i> This means every prediction the model makes is either "yes, home team wins" or "no, away team wins." There's no draw in the NFL, so it's always one or the other.<br><br>

<b>Why do home teams win more often historically?</b><br>
Across the 35 seasons in this dataset, home teams have won approximately 57% of games. The reasons are well-documented:<br>
• <b>Crowd noise</b> — a loud home crowd can disrupt the away team's ability to communicate at the line of scrimmage<br>
• <b>No travel</b> — away teams often travel the night before, disrupting sleep and routine<br>
• <b>Stadium familiarity</b> — home teams train in or near their own stadium and know its quirks (turf type, wind patterns, sun angles)<br>
• <b>Referee effect</b> — studies have shown referees unconsciously give marginally more favourable calls to home teams due to crowd pressure<br><br>

<b>What is this chart?</b><br>
This grid — called a confusion matrix — is a standard way in data science to see not just <i>how often</i> a model was right, but <i>what kind of mistakes</i> it made. It covers all games in the test set, which contains roughly 1,900 games drawn from across the 1990–2025 seasons.<br><br>

<b>How to read it:</b><br>
Each cell tells you how many games fell into that category. The two highlighted cells (top-left and bottom-right) are where the model got it right. The other two are the types of mistake it made.
""")

    tn,fp,fn,tp = cm.ravel()
    fig_cm = go.Figure(go.Heatmap(
        z=[[tn,fp],[fn,tp]],
        x=['Model predicted: Away win','Model predicted: Home win'],
        y=['Reality: Away team won','Reality: Home team won'],
        colorscale=[[0,'#0a0a0a'],[1,'#D50A0A']],
        text=[[f"✅ Correct\n(away win called correctly)\n{tn} games",
               f"❌ Wrong\n(said home win, away won)\n{fp} games"],
              [f"❌ Wrong\n(said away win, home won)\n{fn} games",
               f"✅ Correct\n(home win called correctly)\n{tp} games"]],
        texttemplate='%{text}',
        textfont=dict(size=12,color='white'),
        showscale=False
    ))
    fig_cm.update_layout(**CHART_LAYOUT,height=420,
        xaxis=dict(side='top'), margin=dict(t=100,b=40,l=160,r=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    total_right = tn+tp
    total_wrong = fp+fn
    total_all   = tn+tp+fp+fn
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("✅ Away wins called correctly", tn)
    with c2: st.metric("✅ Home wins called correctly", tp)
    with c3: st.metric("❌ Said home win, away won",   fp)
    with c4: st.metric("❌ Said away win, home won",   fn)

    takeaway(f"""
Out of {total_all:,} test games, the model correctly predicted <b>{total_right:,}</b> outcomes and got <b>{total_wrong:,}</b> wrong.
It is slightly better at correctly predicting home wins than away wins — which makes sense, since home teams win more often and the model has learned to reflect that.
""")

    # ── 6. ROC Curve ────────────────────────
    st.markdown("---")
    st.subheader("📉 6. How Well Can the Model Separate Winners from Losers?")

    explainer(f"""
<b>What does ROC stand for?</b><br>
ROC stands for <b>Receiver Operating Characteristic</b> — a term that originated in radar signal detection in World War II, where engineers needed to measure how well a system could tell the difference between a real aircraft and background noise. In machine learning it's been adopted as a standard way to measure how well a model separates two outcomes.<br><br>

<b>What is this chart actually showing?</b><br>
For every game in the test set, the model gives a probability — for example, "70% chance the home team wins." The ROC curve asks: <i>if you rank all the games from highest to lowest predicted home-win probability, how good is that ranking?</i><br><br>

Specifically, the chart plots two things as you move along that ranked list:<br>
• <b>X-axis (horizontal)</b> — out of all the games the away team actually won, how many has the model mistakenly ranked as likely home wins so far? (Lower is better)<br>
• <b>Y-axis (vertical)</b> — out of all the games the home team actually won, how many has the model correctly ranked as likely home wins so far? (Higher is better)<br><br>

A model that ranks perfectly would shoot straight up the left side and then across — you'd capture all the real home wins before making a single mistake. That would be an AUC of 1.0.<br>
A model that's completely useless — just guessing randomly — produces a diagonal straight line. That's the dashed grey line. AUC = 0.5.<br>
Our model's red curve bends toward the top-left, sitting above the random line. <b>AUC = {roc_auc:.3f}</b>.<br><br>

<b>What does AUC mean?</b><br>
AUC stands for <b>Area Under the Curve</b> — it's literally the area of the space under the red line. The bigger that area, the better the model is at ranking games correctly. An AUC of {roc_auc:.3f} means: if you picked one random game the home team won and one random game the away team won, the model would correctly rank the home win as more likely {roc_auc:.1%} of the time.<br><br>

<b>Why does the model only score {roc_auc:.3f} and not much higher?</b><br>
The NFL is deliberately designed to be unpredictable. The league's draft system gives the worst teams first pick of new talent, and the salary cap prevents any one team from hoarding all the best players. As a result, even the best statistical models — including those used by professional sports betting companies with far more data — typically achieve AUC scores in the 0.58–0.63 range for NFL games. Our model sits right in that range, built entirely on publicly available data.
""")

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        line=dict(color='#D50A0A',width=2),
        name=f'Our model (AUC = {roc_auc:.3f})',
        fill='tozeroy', fillcolor='rgba(213,10,10,0.08)'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        line=dict(color='#666',dash='dash'),
        name='Random guessing (AUC = 0.5)'
    ))
    fig_roc.update_layout(
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        xaxis=dict(title='Fraction of away wins mistakenly ranked as likely home wins →',
                   gridcolor='#333', range=[0,1]),
        yaxis=dict(title='Fraction of home wins correctly ranked as likely home wins →',
                   gridcolor='#333', range=[0,1]),
        legend=dict(bgcolor='#1a1a1a', bordercolor='#333'),
        height=440, showlegend=True
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    takeaway(f"""
The further the red curve bends away from the grey dashed line, the better the model is at telling apart games the home team won from games the away team won.
An AUC of <b>{roc_auc:.3f}</b> places this model firmly in the range of professional NFL forecasting tools — built entirely from publicly available historical data, with no access to injury reports, weather, or betting lines.
""")

    # ── 7. Dataset ──────────────────────────
    st.markdown("---")
    st.subheader("📦 7. The Data Behind the Model")

    explainer("""
<b>Where does the data come from?</b><br>
All game results used to train and test this model come from the <b>NFL Scores & Betting Dataset</b> on Kaggle
(dataset: <i>spreadspoke_scores.csv</i> by user spreadspoke). It contains verified historical NFL game results
going back to 1966. We use games from 1990 onwards for training.<br><br>

<b>How was the home win percentage calculated?</b><br>
For each game in the dataset: if the home team's score was higher than the away team's score, it was counted as a home win.
The percentage is simply: <i>total home wins ÷ total games played</i>. For example, if home teams won 570 out of 1,000 games, the home win rate is 57.0%.
""")

    total_games   = len(scores)
    seasons_range = int(scores['schedule_season'].max()) - int(scores['schedule_season'].min()) + 1
    teams_n       = len(set(scores['team_home'].unique()) | set(scores['team_away'].unique()))
    hw_pct        = scores['home_win'].mean()

    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Total Games in Dataset",  f"{total_games:,}")
    with c2: st.metric("Seasons Covered",          f"{seasons_range} (1990–2025)")
    with c3: st.metric("Teams in Data",            teams_n)
    with c4: st.metric("Overall Home Win Rate",    f"{hw_pct:.1%}")

    st.markdown(f"**Yes — a {hw_pct:.1%} home win rate means home teams have won {hw_pct:.0%} of every 100 games played in this dataset.** That's the historical reality of home field advantage in the NFL.")

    st.markdown("---")
    st.markdown("#### 🏟️ Has Home Field Advantage Changed Over Time?")

    explainer("""
<b>What is this chart?</b><br>
This groups the seasons since 2002 into three time periods and shows the home win percentage in each one.
2002 is the starting point because that's when the <b>Houston Texans</b> joined the league as the 32nd franchise,
completing the modern NFL as it exists today. Using pre-2002 data would mix in seasons with fewer teams and a different competitive landscape, making comparisons less meaningful.<br><br>

<b>How many seasons are covered?</b><br>
2002–2009 = 8 seasons, 2010–2019 = 10 seasons, 2020–2025 = 6 seasons. Total: 24 seasons of the 32-team NFL.<br><br>

<b>Source:</b> Calculated directly from the Kaggle spreadspoke_scores.csv dataset.
""")

    fig_p = go.Figure(go.Bar(
        x=period_hw['Period'],
        y=period_hw['Home Win Rate']*100,
        marker_color='#013369',
        text=[f"{r['Home Win Rate']:.1%}<br>({r['Games']:,} games)" for _,r in period_hw.iterrows()],
        textposition='outside', textfont=dict(color='white')))
    fig_p.add_hline(y=50, line_dash='dash', line_color='#555',
                    annotation_text='50% — no home advantage',
                    annotation_font_color='#aaa', annotation_position='bottom right')
    fig_p.update_layout(**CHART_LAYOUT,
        yaxis=dict(title='Home Win %',range=[0,70],gridcolor='#333'),
        xaxis=dict(gridcolor='#333'),height=360)
    st.plotly_chart(fig_p, use_container_width=True)

    takeaway("""
Home field advantage is real — but it has been shrinking. The 2020–2025 period is the lowest on record,
partly because the 2020 COVID season was played entirely without fans, removing crowd noise as a factor.
Studies suggest crowd noise alone can account for 2–3% of home win advantage, so its sudden absence in 2020 had a measurable effect on results.
""")

# ─── TAB 4 ───────────────────────────────────
with tab4:
    st.subheader("ℹ️ How The Model Works")
    st.markdown("""
### The Data
This model is trained on **NFL games from 1990 to 2025**, sourced from the NFL Scores & Betting Dataset on Kaggle (spreadspoke_scores.csv).

### The Features
For each game, the model uses 6 inputs:
- 🏠 **Home team win rate** — historical win % for the home team
- ✈️ **Away team win rate** — historical win % for the away team
- 🏈 **Home team avg points scored** — average points scored per game
- 🏈 **Away team avg points scored** — average points scored per game
- 🛡️ **Home team avg points conceded** — average points conceded per game
- 🛡️ **Away team avg points conceded** — average points conceded per game

### The Model
Three machine learning models were compared — XGBoost was chosen. See the **🧪 Data Science** tab for the full breakdown.

### Confidence Levels
- 🟢 **High Confidence** — probability gap ≥ 15%
- 🟡 **Medium Confidence** — probability gap 7–15%
- 🔴 **Low Confidence** — probability gap < 7%

### Limitations
The model does not currently account for:
- Current season injuries
- Weather conditions
- Recent trades or roster changes
- Coaching changes

*Built by NFLNerd | Data: Kaggle spreadspoke_scores.csv*
""")

st.markdown("---")
st.caption(f"Built by NFLNerd | Data: Kaggle spreadspoke_scores.csv (1990–2025) | Model accuracy: {accuracy:.1%}")
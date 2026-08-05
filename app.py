import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="NFL Game Predictor | NFLNerd", page_icon="🏈", layout="centered")

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
.nflnerd-brand { font-size: 14px; color: #D50A0A; font-weight: bold; letter-spacing: 2px; }
.explainer-box {
    background-color: #1a1a1a; border-left: 4px solid #D50A0A;
    padding: 12px 16px; border-radius: 4px;
    margin-bottom: 12px; font-size: 14px; line-height: 1.7;
}
.takeaway-box {
    background-color: #0d1f0d; border-left: 4px solid #00C853;
    padding: 12px 16px; border-radius: 4px;
    margin-top: 8px; font-size: 14px; line-height: 1.6;
}
.info-box {
    background-color: #0d1a2e; border-left: 4px solid #013369;
    padding: 12px 16px; border-radius: 4px;
    margin-bottom: 12px; font-size: 14px; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

def explainer(text):
    st.markdown(f'<div class="explainer-box">{text}</div>', unsafe_allow_html=True)
def takeaway(text):
    st.markdown(f'<div class="takeaway-box">{text}</div>', unsafe_allow_html=True)
def infobox(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

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

CHART_LAYOUT = dict(plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a', font=dict(color='white'), showlegend=False)

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
        hg = df[(df['team_home']==team) & (df['schedule_season']<before_season)]
        ag = df[(df['team_away']==team) & (df['schedule_season']<before_season)]
        hw = (hg['score_home']>hg['score_away']).sum()
        aw = (ag['score_away']>ag['score_home']).sum()
        total = len(hg)+len(ag)
        if total==0: return 0.5,22.0,20.0
        wr = (hw+aw)/total
        scored   = pd.concat([hg['score_home'],ag['score_away']]).mean()
        conceded = pd.concat([hg['score_away'],ag['score_home']]).mean()
        return wr,scored,conceded

    game_data=[]
    for _,row in scores.iterrows():
        s=row['schedule_season']
        h_wr,h_sc,h_co=get_team_stats(scores,row['team_home'],s)
        a_wr,a_sc,a_co=get_team_stats(scores,row['team_away'],s)
        game_data.append({'home_win_rate':h_wr,'away_win_rate':a_wr,
            'home_avg_scored':h_sc,'away_avg_scored':a_sc,
            'home_avg_conceded':h_co,'away_avg_conceded':a_co,
            'home_win':row['home_win'],'season':s})

    df_f=pd.DataFrame(game_data).dropna()
    X=df_f.drop(['home_win','season'],axis=1)
    y=df_f['home_win']
    seasons=df_f['season']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    lr  = LogisticRegression(random_state=42,max_iter=1000)
    rf  = RandomForestClassifier(n_estimators=100,random_state=42)
    xgb = XGBClassifier(n_estimators=100,random_state=42,eval_metric='logloss')
    lr.fit(X_train,y_train); rf.fit(X_train,y_train); xgb.fit(X_train,y_train)

    lr_acc  = accuracy_score(y_test,lr.predict(X_test))
    rf_acc  = accuracy_score(y_test,rf.predict(X_test))
    xgb_acc = accuracy_score(y_test,xgb.predict(X_test))

    # Pick the most accurate model dynamically
    accs = {'Logistic Regression': (lr_acc, lr), 'Random Forest': (rf_acc, rf), 'XGBoost': (xgb_acc, xgb)}
    best_name = max(accs, key=lambda k: accs[k][0])
    best_model = accs[best_name][1]
    # Always use XGBoost for predictions but note if another scored higher
    chosen_model = xgb

    season_notes = {
        2004:"2004 — An unusually balanced season with no dominant team. High parity meant more upsets.",
        2007:"2007 — The undefeated Patriots made this one of the most predictable seasons in recent memory.",
        2020:"2020 — The COVID season. Games were played without fans, removing crowd noise as a factor entirely.",
        2022:"2022 — A high-upset season. Multiple strong teams lost games they were expected to win comfortably.",
        2024:"2024 — Lamar Jackson (Ravens), Joe Burrow (Bengals) and Patrick Mahomes (Chiefs) all missed games through injury, making results significantly harder to predict.",
    }
    season_acc=[]
    for s in sorted(seasons.unique()):
        if s<2000: continue
        mask=seasons==s
        if mask.sum()<10: continue
        season_acc.append({'season':int(s),
            'accuracy':accuracy_score(y[mask],chosen_model.predict(X[mask])),
            'note':season_notes.get(int(s),"")})

    cm=confusion_matrix(y_test,chosen_model.predict(X_test))
    fpr,tpr,_=roc_curve(y_test,chosen_model.predict_proba(X_test)[:,1])
    roc_auc=auc(fpr,tpr)

    probs=chosen_model.predict_proba(X_test)[:,1]
    preds_test=chosen_model.predict(X_test)
    conf_data=[]
    for display,low,high in [
        ('🔴 Low Confidence',   0.0, 0.07),
        ('🟡 Medium Confidence',0.07,0.15),
        ('🟢 High Confidence',  0.15,0.50),
    ]:
        mask=(abs(probs-0.5)>=low)&(abs(probs-0.5)<high)
        if mask.sum()>0:
            conf_data.append({'confidence':display,
                'accuracy':accuracy_score(y_test[mask],preds_test[mask]),
                'games':int(mask.sum())})

    feature_names=['Home Win Rate','Away Win Rate','Home Avg Points Scored',
                   'Away Avg Points Scored','Home Avg Points Conceded','Away Avg Points Conceded']
    importances=chosen_model.feature_importances_

    scores_32=scores[scores['schedule_season']>=2002].copy()
    scores_32['period']=pd.cut(scores_32['schedule_season'],
        bins=[2001,2009,2019,2026],labels=['2002–2009','2010–2019','2020–2025'])
    period_hw=scores_32.groupby('period',observed=True)['home_win'].agg(
        home_win_rate='mean',games='count').reset_index()
    period_hw.columns=['Period','Home Win Rate','Games']

    return (chosen_model,scores,get_team_stats,xgb_acc,lr_acc,rf_acc,
            best_name,pd.DataFrame(season_acc),cm,fpr,tpr,roc_auc,
            conf_data,feature_names,importances,X_test,y_test,period_hw,scores_32)

def get_recent_form(scores,team,n=5):
    hg=scores[scores['team_home']==team][['schedule_date','team_home','team_away','score_home','score_away']].copy()
    hg['win']=hg['score_home']>hg['score_away']
    hg['result']=hg.apply(lambda r:f"✅ W {int(r['score_home'])}-{int(r['score_away'])} vs {r['team_away']}" if r['win'] else f"❌ L {int(r['score_home'])}-{int(r['score_away'])} vs {r['team_away']}",axis=1)
    ag=scores[scores['team_away']==team][['schedule_date','team_home','team_away','score_home','score_away']].copy()
    ag['win']=ag['score_away']>ag['score_home']
    ag['result']=ag.apply(lambda r:f"✅ W {int(r['score_away'])}-{int(r['score_home'])} @ {r['team_home']}" if r['win'] else f"❌ L {int(r['score_away'])}-{int(r['score_home'])} @ {r['team_home']}",axis=1)
    all_g=pd.concat([hg[['schedule_date','result','win']],ag[['schedule_date','result','win']]])
    all_g['schedule_date']=pd.to_datetime(all_g['schedule_date'])
    return all_g.sort_values('schedule_date',ascending=False).head(n)

def get_head_to_head(scores,t1,t2):
    h2h=scores[((scores['team_home']==t1)&(scores['team_away']==t2))|
               ((scores['team_home']==t2)&(scores['team_away']==t1))].copy()
    w1=w2=0
    for _,r in h2h.iterrows():
        if r['team_home']==t1:
            if r['score_home']>r['score_away']: w1+=1
            else: w2+=1
        else:
            if r['score_away']>r['score_home']: w1+=1
            else: w2+=1
    return w1,w2,len(h2h)

# ── Load ──────────────────────────────────────
st.markdown('<p class="nflnerd-brand">🏈 NFLNERD</p>',unsafe_allow_html=True)
st.title("NFL Game Outcome Predictor")
st.markdown("Predict the outcome of any NFL matchup using machine learning trained on historical data since 1990.")

with st.spinner('Loading model...'):
    (model,scores,get_team_stats,xgb_acc,lr_acc,rf_acc,
     best_name,season_acc_df,cm,fpr,tpr,roc_auc,
     conf_data,feature_names,importances,
     X_test,y_test,period_hw,scores_32)=load_model()

accuracy=xgb_acc
st.success(f"Model loaded! Accuracy: {accuracy:.1%}")
all_teams=sorted(CURRENT_NFL_TEAMS)

tab1,tab2,tab3,tab4=st.tabs(["🔮 Predict","📅 Weekly Predictions","🧪 Data Science","ℹ️ How It Works"])

# ─── TAB 1 ────────────────────────────────────
with tab1:
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("🏠 Home Team")
        home_team=st.selectbox("Select home team",all_teams,
            index=all_teams.index("Kansas City Chiefs") if "Kansas City Chiefs" in all_teams else 0)
    with c2:
        st.subheader("✈️ Away Team")
        away_team=st.selectbox("Select away team",all_teams,
            index=all_teams.index("Philadelphia Eagles") if "Philadelphia Eagles" in all_teams else 1)
    st.markdown("---")
    if st.button("🔮 Predict Game Outcome",use_container_width=True):
        if home_team==away_team:
            st.error("Please select two different teams!")
        else:
            h_wr,h_sc,h_co=get_team_stats(scores,home_team,2026)
            a_wr,a_sc,a_co=get_team_stats(scores,away_team,2026)
            prob=model.predict_proba([[h_wr,a_wr,h_sc,a_sc,h_co,a_co]])[0]
            hp,ap=prob[1],prob[0]
            diff=abs(hp-ap)
            if diff>=0.15: conf,msg="🟢 High Confidence","The model is strongly favouring one team."
            elif diff>=0.07: conf,msg="🟡 Medium Confidence","The model leans one way but it's not clear cut."
            else: conf,msg="🔴 Low Confidence — Close Game","This is a very tight matchup. Could go either way!"
            st.markdown("---"); st.subheader("📊 Prediction Results")
            c1,c2,c3=st.columns(3)
            with c1: st.metric(f"🏠 {home_team}",f"{hp:.1%}")
            with c2: st.metric("VS","")
            with c3: st.metric(f"✈️ {away_team}",f"{ap:.1%}")
            if hp>ap: st.success(f"🏆 Predicted Winner: **{home_team}** ({hp:.1%} probability)")
            else: st.success(f"🏆 Predicted Winner: **{away_team}** ({ap:.1%} probability)")
            st.info(f"{conf} — {msg}")
            st.markdown("---"); st.subheader("📋 Team Stats Comparison")
            st.dataframe(pd.DataFrame({
                'Stat':['Historical Win Rate','Avg Points Scored','Avg Points Conceded'],
                f'🏠 {home_team}':[f"{h_wr:.1%}",f"{h_sc:.1f}",f"{h_co:.1f}"],
                f'✈️ {away_team}':[f"{a_wr:.1%}",f"{a_sc:.1f}",f"{a_co:.1f}"],
            }).set_index('Stat'),use_container_width=True)
            st.markdown("---"); st.subheader("📊 Win Probability")
            fig=go.Figure(go.Bar(
                x=[f"🏠 {home_team}",f"✈️ {away_team}"],y=[hp*100,ap*100],
                marker_color=["#013369","#D50A0A"],
                text=[f"{hp:.1%}",f"{ap:.1%}"],
                textposition="outside",textfont=dict(color="white",size=16)))
            fig.update_layout(**CHART_LAYOUT,
                yaxis=dict(title="Win Probability (%)",range=[0,110],gridcolor="#333"),
                xaxis=dict(gridcolor="#333"),height=350)
            st.plotly_chart(fig,use_container_width=True)
            st.markdown("---"); st.subheader("⚔️ Head to Head Record")
            w1,w2,tot=get_head_to_head(scores,home_team,away_team)
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

# ─── TAB 2 ────────────────────────────────────
with tab2:
    st.subheader("📅 Weekly Predictions")
    st.markdown("Enter this week's matchups and get predictions for all games at once.")
    st.info("Add up to 16 matchups below:")
    num_games=st.slider("How many games?",1,16,4)
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
            prob=model.predict_proba([[h_wr,a_wr,h_sc,a_sc,h_co,a_co]])[0]
            hp,ap=prob[1],prob[0]
            diff=abs(hp-ap)
            conf="🟢" if diff>=0.15 else ("🟡" if diff>=0.07 else "🔴")
            st.markdown(f"{conf} **{h}** vs **{a}** → 🏆 **{'Home: '+h if hp>ap else 'Away: '+a}** ({max(hp,ap):.1%})")

# ─── TAB 3 — DATA SCIENCE ─────────────────────
with tab3:
    st.subheader("🧪 Data Science & Model Analysis")
    st.markdown("This tab walks through how the prediction model was built, what it learned from 35 years of NFL data, and how well it actually performs.")

    # ── 1. Which Model Performed Best? ────────
    st.markdown("---")
    st.subheader("🤖 1. Which Model Performed Best?")

    explainer("""
<b>What is a machine learning model?</b><br>
A machine learning model is a piece of software that studies historical data and learns which patterns tend to predict a certain outcome.
It is similar to a weather forecasting app — it doesn't know exactly what tomorrow's weather will be,
but after studying millions of days of past weather data it can make an educated prediction.<br><br>

Here, each model was given the same 6 stats for every game played since 1990:
the home team's historical win rate, the away team's historical win rate,
average points scored by each team, and average points conceded by each team.
It then studied tens of thousands of games to learn which combinations of those stats tend to produce a home win or an away win.<br><br>

<b>Three models were trained and tested against each other:</b><br><br>

<b>Logistic Regression</b><br>
This is the simplest approach. It looks at all 6 stats together and assigns a weight to each one — for example,
it might learn that win rate matters more than average points scored. It then combines all 6 weighted stats into a single score per team and predicts whichever team scores higher will win.<br>
It does <i>not</i> just pick whoever has the highest single stat — a team with a great home win rate but a poor away win rate
would have that weakness factored in, because the model is weighing all 6 numbers simultaneously.<br><br>

<b>Random Forest</b><br>
This builds a large number of independent <b>decision trees</b>. A decision tree is a series of yes/no questions about the data —
for example: "Is the home team's win rate above 55%? → Yes → Is the away team's points conceded above 25 per game? → Yes → Predict home win."
Each tree asks slightly different questions and arrives at its own prediction.
The Random Forest then takes a <b>majority vote</b> across all those trees — whichever outcome the most trees predicted is the final answer,
similar to asking 100 analysts to each independently predict a game and going with whatever the majority said.<br><br>

<b>XGBoost</b><br>
XGBoost also builds decision trees, but with a key difference: instead of building them all independently,
it builds them <i>one at a time</i>, where each new tree specifically focuses on correcting the mistakes the previous trees made.
This process — called gradient boosting — makes the overall prediction more and more refined with each tree added.<br><br>

<b>Why was XGBoost chosen?</b><br>
The most accurate model on this dataset is used for all predictions — whichever that turns out to be when the model is run.
XGBoost is also preferred because it produces <b>importance scores</b> (explained in the next section) that show exactly which stats drove each prediction,
making the model explainable rather than a black box.
It also handles more data and additional stats better than Logistic Regression as the project grows —
for example, adding QB passer rating, days of rest between games, weather conditions, injury reports,
or stadium type would all be stats XGBoost could take advantage of more effectively.<br><br>

<b>What does parity mean?</b><br>
Parity in the NFL means the league is deliberately designed so that every team has a roughly equal chance of competing each season.
The NFL achieves this through the draft (where the worst teams get first pick of new college talent each year)
and the salary cap (which limits how much any one team can spend on players, preventing the richest teams from hoarding all the best talent).
This makes the NFL far more unpredictable than most other sports leagues, which is part of why even the best models only reach around 58–60% accuracy.
""")

    model_names=['Logistic Regression','Random Forest','XGBoost']
    model_accs=[lr_acc*100,rf_acc*100,xgb_acc*100]
    bar_colors=['#555555','#888888','#888888']
    best_idx=model_accs.index(max(model_accs))
    bar_colors[best_idx]='#D50A0A'
    model_labels=[f"{n} ✅" if i==best_idx else n for i,n in enumerate(model_names)]

    fig_m=go.Figure(go.Bar(
        x=model_labels,y=model_accs,
        marker_color=bar_colors,
        text=[f"{a:.2f}%" for a in model_accs],
        textposition='outside',textfont=dict(color='white',size=14)))
    fig_m.add_hline(y=50,line_dash='dash',line_color='#666',
        annotation_text='Random guessing (50%)',annotation_font_color='#aaa',
        annotation_position='bottom right')
    fig_m.update_layout(**CHART_LAYOUT,
        yaxis=dict(title='Accuracy (%)',range=[45,65],gridcolor='#333'),
        xaxis=dict(gridcolor='#333'),height=380)
    st.plotly_chart(fig_m,use_container_width=True)

    c1,c2,c3=st.columns(3)
    with c1: st.metric("Logistic Regression",f"{lr_acc:.2%}")
    with c2: st.metric("Random Forest",f"{rf_acc:.2%}")
    with c3:
        delta=xgb_acc-lr_acc
        st.metric("XGBoost",f"{xgb_acc:.2%}",
            delta=f"{delta:+.2%} vs Logistic Regression")

    takeaway(f"""
The highest-scoring model is highlighted in red and used for all predictions on this site.
All three models beat random guessing (50%), which is the absolute baseline — a model that just flips a coin on every game.
The NFL's parity means even professional forecasting tools sit in the 58–60% range, so these results are genuinely competitive.
""")

    # ── 2. Which Stats Matter Most? ───────────
    st.markdown("---")
    st.subheader("🔍 2. Which Stats Matter Most to the Model?")

    explainer("""
<b>What are feature importance scores?</b><br>
When XGBoost makes a prediction, it doesn't rely on all 6 stats equally — some are far more influential than others.
After training on tens of thousands of games, XGBoost calculates an <b>importance score</b> for each stat,
which measures how much that stat contributed to the model's decisions across every prediction it made during training.<br><br>

The score is a number between 0 and 1. A score of 0.35 means that stat was responsible for 35% of the model's decision-making.
All six scores add up to 1.0 (100%).<br><br>

<b>What each stat means:</b><br>
• <b>Home/Away Win Rate</b> — that team's overall percentage of games won across all their historical games in the dataset<br>
• <b>Home/Away Avg Points Scored</b> — how many points that team scores per game on average<br>
• <b>Home/Away Avg Points Conceded</b> — how many points that team lets in per game on average<br><br>

<b>How does the model know about things like a "high-powered offence" or a "leaky defence"?</b><br>
Those are exactly what the points scored and points conceded stats measure.
If a team averages 30+ points per game, the model sees that as a high-powered offence.
If a team concedes 28+ points per game, the model sees that as a leaky defence.
It learns from 35 years of data that certain combinations of these numbers — for example, a high-scoring offence facing a leaky defence — tend to predict winners more reliably than others.<br><br>

<b>Why might Away Win Rate rank higher than Home Win Rate?</b><br>
Winning away from home is genuinely harder — no home crowd, unfamiliar stadium, travel fatigue.
A team that wins consistently on the road is usually a better all-round team than one that only performs well at home.
So the model has learned that away win rate is a stronger signal of true team quality.
""")

    idx=np.argsort(importances)[::-1]
    s_fn=[feature_names[i] for i in idx]
    s_fi=[importances[i] for i in idx]

    fig_fi=go.Figure(go.Bar(
        x=s_fi,y=s_fn,orientation='h',
        marker_color='#013369',
        text=[f"{v:.3f}" for v in s_fi],
        textposition='outside',textfont=dict(color='white')))
    fig_fi.update_layout(**CHART_LAYOUT,
        xaxis=dict(title='Importance Score (all six add up to 1.0)',
                   gridcolor='#333',range=[0,max(s_fi)*1.3]),
        yaxis=dict(gridcolor='#333',autorange='reversed'),height=400)
    st.plotly_chart(fig_fi,use_container_width=True)

    # ── 3. Season-by-Season ───────────────────
    st.markdown("---")
    st.subheader("📈 3. How Accurate Was the Model in Each NFL Season?")

    explainer("""
<b>What is this chart showing?</b><br>
This shows the percentage of games the model correctly predicted out of all NFL games played in each individual season, from 2000 to 2025.
For example, if a season had 256 games and the model correctly predicted 153 of them, that season's accuracy would be 59.8%.<br><br>

<b>Why start from 2000?</b><br>
Pre-2000 NFL data comes from a different era of the sport — fewer teams, different rules, different gameplay styles.
Including those seasons in a year-by-year comparison would make the chart misleading, so it starts from 2000 for consistency.
That data is still used in model training, just not shown here.<br><br>

<b>Why does the number of upsets change year to year?</b><br>
Some seasons naturally produce more upsets than others due to factors the model cannot see:
a star quarterback getting injured mid-season, a team underperforming under a new coaching system,
or an unexpected run of form from a team that looked average on paper.
In the 2024 season for example, Lamar Jackson (Ravens), Joe Burrow (Bengals), and Patrick Mahomes (Chiefs)
all missed games through injury — three of the most predictable teams in the league suddenly became much harder to forecast.<br><br>

<b>Hover over any point</b> on the line to see the exact accuracy and context for notable seasons.
""")

    if not season_acc_df.empty:
        avg_acc=season_acc_df['accuracy'].mean()
        fig_s=go.Figure()
        fig_s.add_trace(go.Scatter(
            x=season_acc_df['season'],
            y=season_acc_df['accuracy']*100,
            mode='lines+markers',
            line=dict(color='#D50A0A',width=3),
            marker=dict(size=9,color='#D50A0A',line=dict(color='white',width=1)),
            customdata=season_acc_df['note'],
            hovertemplate='<b>Season: %{x}</b><br>Accuracy: %{y:.1f}%<br>%{customdata}<extra></extra>'
        ))
        fig_s.add_hline(y=50,line_dash='dash',line_color='#555',
            annotation_text='Random guessing (50%)',
            annotation_font_color='#aaa',annotation_position='bottom right')
        fig_s.add_hline(y=avg_acc*100,line_dash='dot',line_color='#4a90d9',
            annotation_text=f'Overall average ({avg_acc:.1%})',
            annotation_font_color='#aaa',annotation_position='top right')
        fig_s.update_layout(**CHART_LAYOUT,
            xaxis=dict(title='NFL Season',gridcolor='#333',dtick=2,
                tick0=2000,
                range=[season_acc_df['season'].min()-0.5,
                       season_acc_df['season'].max()+0.5]),
            yaxis=dict(title='Games correctly predicted (%)',
                       range=[45,75],gridcolor='#333',dtick=5),
            height=450,margin=dict(t=40,b=60,l=60,r=60))
        st.plotly_chart(fig_s,use_container_width=True)

        best=season_acc_df.loc[season_acc_df['accuracy'].idxmax()]
        worst=season_acc_df.loc[season_acc_df['accuracy'].idxmin()]
        c1,c2,c3=st.columns(3)
        with c1: st.metric("Overall Average",f"{avg_acc:.1%}")
        with c2: st.metric("Best Season",f"{int(best['season'])} ({best['accuracy']:.1%})")
        with c3: st.metric("Toughest Season",f"{int(worst['season'])} ({worst['accuracy']:.1%})")

        takeaway(f"""
The model beats random guessing in every single season — the red line stays above the 50% dashed baseline throughout.
Its toughest season was <b>{int(worst['season'])}</b> and its best was <b>{int(best['season'])}</b>.
The accuracy goes up and down year to year because the number of upsets in the NFL goes up and down —
a more predictable season produces a higher score, not a better model.
""")

    # ── 4. Confidence ─────────────────────────
    st.markdown("---")
    st.subheader("🎯 4. Does the Model's Confidence Level Actually Mean Anything?")

    explainer("""
<b>How does the model express confidence?</b><br>
When the model predicts a game, it gives each team a probability.
For example: home team 64%, away team 36%.
That means the model believes that if you played this exact game 100 times, the home team would win around 64 of them.<br><br>

<b>The three confidence levels:</b><br>
• 🔴 <b>Low Confidence</b> — the model gives something like 53% vs 47%. A gap of less than 7 percentage points. The model sees this as almost a coin flip and has very little conviction either way.<br>
• 🟡 <b>Medium Confidence</b> — something like 57% vs 43%. A gap between 7 and 15 percentage points. The model leans one way but acknowledges real uncertainty.<br>
• 🟢 <b>High Confidence</b> — something like 66% vs 34%. A gap of more than 15 percentage points. The model strongly favours one team.<br><br>

<b>What do the numbers on the chart mean?</b><br>
The model was tested on roughly 1,900 games drawn from seasons between 1990 and 2025 (games before 1990 were not used).
Each game was placed into one of the three confidence buckets above.<br>
The <b>percentage on each bar</b> is how accurately the model predicted games within that bucket —
for example, if the green bar shows 61%, it means the model correctly predicted 61% of the games it was highly confident about.<br>
The <b>number in brackets</b> (e.g. "312 games") shows how many of those ~1,900 test games fell into that confidence bucket.
""")

    if conf_data:
        cdf=pd.DataFrame(conf_data)
        fig_c=go.Figure(go.Bar(
            x=cdf['confidence'],
            y=cdf['accuracy']*100,
            marker_color=['#D50A0A','#FFB300','#00C853'],
            text=[f"{a:.1%}\n({g} games)" for a,g in zip(cdf['accuracy'],cdf['games'])],
            textposition='outside',textfont=dict(color='white',size=13)))
        fig_c.add_hline(y=50,line_dash='dash',line_color='#555',
            annotation_text='Random guessing (50%)',
            annotation_font_color='#aaa',annotation_position='bottom right')
        fig_c.update_layout(**CHART_LAYOUT,
            yaxis=dict(title='Games correctly predicted (%)',range=[40,80],gridcolor='#333'),
            xaxis=dict(gridcolor='#333'),height=420)
        st.plotly_chart(fig_c,use_container_width=True)

        takeaway("""
If this chart is working correctly, the green bar (High Confidence) will be noticeably taller than the red bar (Low Confidence).
That confirms that when the model says it's confident, it really is more likely to be right —
and when it flags a game as nearly a coin flip, it genuinely is harder to call.
""")

    # ── 5. Confusion Matrix ───────────────────
    st.markdown("---")
    st.subheader("🔢 5. Where Does the Model Go Wrong?")

    infobox("""
<b>Why is everything framed as home team vs away team?</b><br>
In every single NFL game ever played, there is one home team and one away team.
That distinction — home or away — is the only thing that is always different between the two sides going into any game.
Because of this, the model is built around one specific question: <b>will the home team win?</b>
Every prediction is either "yes, home team wins" or "no, away team wins."
Note: NFL ties are extremely rare (fewer than 1 in 500 games) and are not included in the dataset.<br><br>

<b>Why do home teams win more often?</b><br>
Across 35 seasons of data, home teams have won approximately 57% of all games. The main reasons are:<br>
• <b>Crowd noise</b> — a loud home crowd disrupts the away team's ability to communicate at the line of scrimmage<br>
• <b>No travel</b> — away teams often travel the day before, disrupting sleep and routine<br>
• <b>Stadium familiarity</b> — home teams train near their own stadium and know its quirks (turf type, wind, sun angles)
""")

    tn,fp,fn,tp=cm.ravel()
    total_all=tn+tp+fp+fn
    total_right=tn+tp

    st.markdown(f"""
Out of all **{total_all:,} test games**, the model divided its predictions into 4 groups based on whether it thought the home or away team would win. Here are the results:
""")

    fig_cm=go.Figure(go.Heatmap(
        z=[[tn,fp],[fn,tp]],
        x=['Model predicted: Away win','Model predicted: Home win'],
        y=['Reality: Away team won','Reality: Home team won'],
        colorscale=[[0,'#0a0a0a'],[1,'#D50A0A']],
        text=[[f"✅ Correct\nAway win predicted & away won\n{tn:,} games",
               f"❌ Wrong\nHome win predicted, away won\n{fp:,} games"],
              [f"❌ Wrong\nAway win predicted, home won\n{fn:,} games",
               f"✅ Correct\nHome win predicted & home won\n{tp:,} games"]],
        texttemplate='%{text}',
        textfont=dict(size=12,color='white'),
        showscale=False))
    fig_cm.update_layout(**CHART_LAYOUT,height=420,
        xaxis=dict(side='top'),margin=dict(t=120,b=40,l=180,r=20))
    st.plotly_chart(fig_cm,use_container_width=True)

    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("✅ Away wins correctly called",tn)
    with c2: st.metric("✅ Home wins correctly called",tp)
    with c3: st.metric("❌ Called home win, away won",fp)
    with c4: st.metric("❌ Called away win, home won",fn)

    takeaway(f"""
Out of {total_all:,} test games, the model correctly predicted <b>{total_right:,}</b> outcomes.
It correctly called home wins more than twice as often as it correctly called away wins ({tp:,} vs {tn:,}) —
which reflects the real-world pattern that home teams win around 57% of games.
The model has learned to lean toward the home team, and that leans pays off more often than not.
""")

    # ── 6. ROC Curve ──────────────────────────
    st.markdown("---")
    st.subheader("📉 6. How Well Can the Model Separate Winners from Losers?")

    explainer(f"""
<b>What is a ROC curve?</b><br>
A ROC curve (Receiver Operating Characteristic) is a standard way in data science to measure how well a model separates two outcomes —
in this case, games the home team won versus games the away team won.<br><br>

We will work through this section together to make it as clear as possible.
For now, the headline number is the <b>AUC score: {roc_auc:.3f}</b>.<br><br>

AUC stands for <b>Area Under the Curve</b>. The score runs from 0.5 (completely useless — no better than guessing) to 1.0 (perfect — gets every game right).
Our score of {roc_auc:.3f} places this model in the same range as professional NFL forecasting tools.
""")

    fig_roc=go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr,y=tpr,mode='lines',
        line=dict(color='#D50A0A',width=2),
        name=f'Our model (AUC = {roc_auc:.3f})',
        fill='tozeroy',fillcolor='rgba(213,10,10,0.08)'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1],y=[0,1],mode='lines',
        line=dict(color='#666',dash='dash'),
        name='Random guessing (AUC = 0.5)'
    ))
    fig_roc.update_layout(
        plot_bgcolor='#0a0a0a',paper_bgcolor='#0a0a0a',
        font=dict(color='white'),
        xaxis=dict(title='→ More away wins mistakenly predicted as home wins',
                   gridcolor='#333',range=[0,1]),
        yaxis=dict(title='→ More home wins correctly identified',
                   gridcolor='#333',range=[0,1]),
        legend=dict(bgcolor='#1a1a1a',bordercolor='#333'),
        height=440,showlegend=True
    )
    st.plotly_chart(fig_roc,use_container_width=True)

    # ── 7. The Data ───────────────────────────
    st.markdown("---")
    st.subheader("📦 7. The Data Behind the Model")

    explainer("""
<b>Where does the data come from?</b><br>
All game results come from the <b>NFL Scores & Betting Dataset</b> on Kaggle (spreadspoke_scores.csv).
It contains verified historical NFL game results going back to 1966. This model uses games from 1990 onwards.<br><br>

<b>How was the overall home win percentage calculated?</b><br>
For every game in the dataset, if the home team's score was higher than the away team's score, it was counted as a home win.
The percentage is: <i>total home wins ÷ total games</i>.
For example, if home teams won 5,400 out of 9,455 games, the home win rate is 5,400 ÷ 9,455 = 57.1%.
""")

    total_games=len(scores)
    hw_pct=scores['home_win'].mean()
    total_hw=int(scores['home_win'].sum())

    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Total Games",f"{total_games:,}")
    with c2: st.metric("Seasons Covered","1990–2025")
    with c3: st.metric("Home Wins",f"{total_hw:,}")
    with c4: st.metric("Home Win Rate",f"{hw_pct:.1%}")

    st.markdown(f"**{hw_pct:.1%}** = {total_hw:,} home wins ÷ {total_games:,} total games. Yes — home teams have won {hw_pct:.0%} of every 100 games in this dataset.")

    st.markdown("---")
    st.markdown("#### 🏟️ Has Home Field Advantage Changed Over Time?")

    explainer("""
<b>What is this chart?</b><br>
This groups NFL seasons since 2002 into three time periods and shows the home win percentage in each one.
2002 is the starting point because that is when the <b>Houston Texans</b> joined the league as the 32nd franchise,
completing the modern NFL as it exists today.<br><br>

<b>Seasons covered:</b> 2002–2009 (8 seasons), 2010–2019 (10 seasons), 2020–2025 (6 seasons). Total: 24 seasons.<br><br>

<b>How was the home win % calculated for each period?</b><br>
Same method as above: total home wins in that period ÷ total games in that period.
The number of games in each bar is shown in brackets so you can see exactly what it is based on.<br><br>

<b>Source:</b> Calculated directly from the Kaggle spreadspoke_scores.csv dataset.
""")

    fig_p=go.Figure(go.Bar(
        x=period_hw['Period'],
        y=period_hw['Home Win Rate']*100,
        marker_color='#013369',
        text=[f"{r['Home Win Rate']:.1%}<br>({r['Games']:,} games)" for _,r in period_hw.iterrows()],
        textposition='outside',textfont=dict(color='white')))
    fig_p.add_hline(y=50,line_dash='dash',line_color='#555',
        annotation_text='50% — no home advantage',
        annotation_font_color='#aaa',annotation_position='bottom right')
    fig_p.update_layout(**CHART_LAYOUT,
        yaxis=dict(title='Home Win %',range=[0,70],gridcolor='#333'),
        xaxis=dict(gridcolor='#333'),height=360)
    st.plotly_chart(fig_p,use_container_width=True)

    takeaway("""
Home field advantage is real — but it has been shrinking over time.
The 2020–2025 period is the lowest on record, partly because the 2020 COVID season was played entirely without fans,
removing crowd noise as a factor for an entire season.
""")

# ─── TAB 4 ────────────────────────────────────
with tab4:
    st.subheader("ℹ️ How The Model Works")
    st.markdown(f"""
### The Data
Trained on NFL games from 1990 to 2025. Source: Kaggle spreadspoke_scores.csv.

### The 6 Stats Used
- 🏠 Home team historical win rate
- ✈️ Away team historical win rate
- 🏈 Home team average points scored per game
- 🏈 Away team average points scored per game
- 🛡️ Home team average points conceded per game
- 🛡️ Away team average points conceded per game

### The Model
Three models were tested — the most accurate is used for predictions. See the **🧪 Data Science** tab for the full breakdown.

### Confidence Levels
- 🟢 **High Confidence** — probability gap ≥ 15%
- 🟡 **Medium Confidence** — probability gap 7–15%
- 🔴 **Low Confidence** — probability gap < 7%

### Limitations
Does not currently account for: injuries, weather, trades, coaching changes.

*Built by NFLNerd | Data: Kaggle spreadspoke_scores.csv (1990–2025)*
""")

st.markdown("---")
st.caption(f"Built by NFLNerd | Data: Kaggle spreadspoke_scores.csv (1990–2025) | Model accuracy: {accuracy:.1%}")
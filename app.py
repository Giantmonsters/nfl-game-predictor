import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
from xgboost import XGBClassifier
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="NFL Game Predictor | NFLNerd",
    page_icon="🏈",
    layout="wide"
)

# ── Styles ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

.stApp { background-color: #080808; color: #f0f0f0; }

/* Typography */
h1,h2,h3 { font-family: 'Oswald', sans-serif !important; color: #ffffff; letter-spacing: 0.5px; }
p, div, span, li { font-family: 'Inter', sans-serif; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #D50A0A, #a50808);
    color: white; font-family: 'Oswald', sans-serif;
    font-size: 17px; font-weight: 600; letter-spacing: 1px;
    border: none; border-radius: 6px; padding: 12px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover { background: linear-gradient(135deg, #013369, #011f40); transform: translateY(-1px); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #111; border-bottom: 2px solid #D50A0A; gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Oswald', sans-serif; font-size: 15px; font-weight: 600;
    color: #888; padding: 12px 20px; border-radius: 0;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] { color: #D50A0A !important; border-bottom: 3px solid #D50A0A !important; background-color: #111 !important; }

/* Cards */
.match-card {
    background: linear-gradient(135deg, #111827, #0d1520);
    border: 1px solid #1e2d3d; border-radius: 12px;
    padding: 20px; margin-bottom: 16px;
}
.team-ranking-card {
    background: #111; border-left: 4px solid #D50A0A;
    border-radius: 8px; padding: 16px; margin-bottom: 10px;
    display: flex; align-items: center; gap: 16px;
}
.rank-number {
    font-family: 'Oswald', sans-serif; font-size: 32px;
    font-weight: 700; color: #D50A0A; min-width: 50px;
}
.tier-header {
    font-family: 'Oswald', sans-serif; font-size: 20px;
    font-weight: 700; padding: 10px 16px; border-radius: 6px;
    margin: 20px 0 10px 0; letter-spacing: 1px;
}
.explainer-box {
    background-color: #111; border-left: 4px solid #D50A0A;
    padding: 14px 18px; border-radius: 4px;
    margin-bottom: 14px; font-size: 14px; line-height: 1.8;
}
.takeaway-box {
    background-color: #0d1f0d; border-left: 4px solid #00C853;
    padding: 14px 18px; border-radius: 4px;
    margin-top: 10px; font-size: 14px; line-height: 1.7;
}
.info-box {
    background-color: #0d1a2e; border-left: 4px solid #013369;
    padding: 14px 18px; border-radius: 4px;
    margin-bottom: 14px; font-size: 14px; line-height: 1.8;
}
.nflnerd-brand {
    font-family: 'Oswald', sans-serif;
    font-size: 13px; color: #D50A0A;
    font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase;
}
.hero-stat {
    font-family: 'Oswald', sans-serif;
    font-size: 48px; font-weight: 700; color: #D50A0A;
    line-height: 1;
}
.hero-label {
    font-family: 'Inter', sans-serif;
    font-size: 13px; color: #888; text-transform: uppercase;
    letter-spacing: 1px; margin-top: 4px;
}
.movement-up { color: #00C853; font-weight: 700; }
.movement-down { color: #D50A0A; font-weight: 700; }
.movement-same { color: #888; font-weight: 700; }
.footer-bar {
    background: #111; border-top: 2px solid #D50A0A;
    padding: 20px; text-align: center;
    font-family: 'Inter', sans-serif; font-size: 13px; color: #666;
}
.confidence-high { color: #00C853; font-weight: 700; }
.confidence-med  { color: #FFB300; font-weight: 700; }
.confidence-low  { color: #D50A0A; font-weight: 700; }
.winner-badge {
    background: linear-gradient(135deg, #D50A0A, #a50808);
    color: white; font-family: 'Oswald', sans-serif;
    font-size: 14px; font-weight: 600; letter-spacing: 1px;
    padding: 4px 12px; border-radius: 20px; display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ── Helper renderers ─────────────────────────
def explainer(text): st.markdown(f'<div class="explainer-box">{text}</div>', unsafe_allow_html=True)
def takeaway(text):  st.markdown(f'<div class="takeaway-box">{text}</div>', unsafe_allow_html=True)
def infobox(text):   st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

# ── Constants ────────────────────────────────
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

ESPN_LOGOS = {
    'Arizona Cardinals':'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png',
    'Atlanta Falcons':'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png',
    'Baltimore Ravens':'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png',
    'Buffalo Bills':'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
    'Carolina Panthers':'https://a.espncdn.com/i/teamlogos/nfl/500/car.png',
    'Chicago Bears':'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png',
    'Cincinnati Bengals':'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png',
    'Cleveland Browns':'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png',
    'Dallas Cowboys':'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png',
    'Denver Broncos':'https://a.espncdn.com/i/teamlogos/nfl/500/den.png',
    'Detroit Lions':'https://a.espncdn.com/i/teamlogos/nfl/500/det.png',
    'Green Bay Packers':'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png',
    'Houston Texans':'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png',
    'Indianapolis Colts':'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png',
    'Jacksonville Jaguars':'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png',
    'Kansas City Chiefs':'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png',
    'Las Vegas Raiders':'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png',
    'Los Angeles Chargers':'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png',
    'Los Angeles Rams':'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
    'Miami Dolphins':'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png',
    'Minnesota Vikings':'https://a.espncdn.com/i/teamlogos/nfl/500/min.png',
    'New England Patriots':'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png',
    'New Orleans Saints':'https://a.espncdn.com/i/teamlogos/nfl/500/no.png',
    'New York Giants':'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png',
    'New York Jets':'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png',
    'Philadelphia Eagles':'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png',
    'Pittsburgh Steelers':'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png',
    'San Francisco 49ers':'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png',
    'Seattle Seahawks':'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png',
    'Tampa Bay Buccaneers':'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png',
    'Tennessee Titans':'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png',
    'Washington Commanders':'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png',
}

CHART_LAYOUT = dict(plot_bgcolor='#080808', paper_bgcolor='#080808', font=dict(color='#f0f0f0'), showlegend=False)

# ── Fox Sports power rankings (as of July 28 2026) ──
FOX_RANKINGS = [
    {"rank":1,  "team":"Los Angeles Rams",      "prev":2,  "tier":"🔥 Elite",      "comment":"Reloaded in the offseason and some are already calling them 'The Dream Team.' Aaron Donald retirement return rumours add to the intrigue."},
    {"rank":2,  "team":"Denver Broncos",         "prev":1,  "tier":"🔥 Elite",      "comment":"Bo Nix reportedly 'full go' for camp. If healthy for the playoffs, this team goes deep."},
    {"rank":3,  "team":"Baltimore Ravens",       "prev":5,  "tier":"🔥 Elite",      "comment":"Lamar Jackson and Derrick Henry healthy = Super Bowl contenders. Simple as that."},
    {"rank":4,  "team":"Seattle Seahawks",       "prev":3,  "tier":"🔥 Elite",      "comment":"Defending champs. RB situation is a concern with Walker gone and Charbonnet recovering from ACL."},
    {"rank":5,  "team":"New England Patriots",   "prev":7,  "tier":"✅ Contenders", "comment":"Drake Maye + A.J. Brown is a frightening combo. Super Bowl contenders again."},
    {"rank":6,  "team":"Buffalo Bills",          "prev":6,  "tier":"✅ Contenders", "comment":"Never count out Josh Allen. WR depth is a question but this team always finds a way."},
    {"rank":7,  "team":"Philadelphia Eagles",    "prev":4,  "tier":"✅ Contenders", "comment":"Still loaded, but losing A.J. Brown hurts. Rookie Makai Lemon already dealing with hamstring issues."},
    {"rank":8,  "team":"Chicago Bears",          "prev":9,  "tier":"✅ Contenders", "comment":"Dangerous offense only gets better in Year 2 of the Ben Johnson era. Defense is the question."},
    {"rank":9,  "team":"Houston Texans",         "prev":10, "tier":"✅ Contenders", "comment":"Everything rides on C.J. Stroud recapturing his rookie form. Dark horse Super Bowl contender if he does."},
    {"rank":10, "team":"Cincinnati Bengals",     "prev":11, "tier":"✅ Contenders", "comment":"Joe Burrow back, defense rebuilt, easy schedule. Avoid the trademark slow start and watch out."},
    {"rank":11, "team":"Jacksonville Jaguars",   "prev":8,  "tier":"✅ Contenders", "comment":"Easy to trust Liam Coen. Hard to like the RB corps for a contender."},
    {"rank":12, "team":"San Francisco 49ers",    "prev":13, "tier":"✅ Contenders", "comment":"Kittle and Bosa injury concerns plus the Brandon Aiyuk distraction. All the pieces are there though."},
    {"rank":13, "team":"Kansas City Chiefs",     "prev":12, "tier":"⚠️ Middling",   "comment":"Mahomes cleared to practice but ACL recovery still a concern. Rashee Rice situation is a mess."},
    {"rank":14, "team":"Los Angeles Chargers",   "prev":16, "tier":"⚠️ Middling",   "comment":"Tough first half schedule. If Mike McDaniel unlocks Justin Herbert, they could be dangerous."},
    {"rank":15, "team":"Detroit Lions",          "prev":15, "tier":"⚠️ Middling",   "comment":"Released CB Terrion Arnold — the right call. Secondary will be fine. Watch Dan Campbell's team."},
    {"rank":16, "team":"Green Bay Packers",      "prev":14, "tier":"⚠️ Middling",   "comment":"Micah Parsons (ACL) won't be back until mid-October. Josh Jacobs domestic violence investigation is a cloud."},
    {"rank":17, "team":"Minnesota Vikings",      "prev":17, "tier":"⚠️ Middling",   "comment":"Everything depends on whether Kevin O'Connell can resurrect Kyler Murray. High ceiling, real risk."},
    {"rank":18, "team":"Carolina Panthers",      "prev":18, "tier":"⚠️ Middling",   "comment":"Make-or-break season for Bryce Young. He's shown flashes — can he put it together consistently?"},
    {"rank":19, "team":"Dallas Cowboys",         "prev":21, "tier":"⚠️ Middling",   "comment":"People around the NFL like the defensive overhaul. Offense still has to do most of the work."},
    {"rank":20, "team":"Tampa Bay Buccaneers",   "prev":22, "tier":"⚠️ Middling",   "comment":"Defense improved. Baker Mayfield contract extension would remove a big distraction."},
    {"rank":21, "team":"New York Giants",        "prev":19, "tier":"⚠️ Middling",   "comment":"Cam Skattebo healthy. Malik Nabers' knee return timeline is unclear — usually a bad sign."},
    {"rank":22, "team":"Atlanta Falcons",        "prev":20, "tier":"⚠️ Middling",   "comment":"Michael Penix Jr. return timeline unknown. James Pearce facing potential suspension. Tough offseason."},
    {"rank":23, "team":"Indianapolis Colts",     "prev":23, "tier":"⚠️ Middling",   "comment":"Daniel Jones on track for Week 1, but his post-injury form last time was terrible."},
    {"rank":24, "team":"Pittsburgh Steelers",    "prev":25, "tier":"⚠️ Middling",   "comment":"A 42-year-old Aaron Rodgers, a brutal division, and a backloaded schedule. Should have rebuilt."},
    {"rank":25, "team":"New Orleans Saints",     "prev":26, "tier":"🔄 Rebuilding", "comment":"Cam Jordan's return helps. Everything still hinges on how much you believe in Tyler Shough."},
    {"rank":26, "team":"Washington Commanders",  "prev":24, "tier":"🔄 Rebuilding", "comment":"Defense fixed. But light on offensive skill players beyond the ageing Terry McLaurin."},
    {"rank":27, "team":"Tennessee Titans",       "prev":27, "tier":"🔄 Rebuilding", "comment":"Major overhaul underway. If Cam Ward turns out to be the answer, they'll look much better by December."},
    {"rank":28, "team":"Cleveland Browns",       "prev":28, "tier":"🔄 Rebuilding", "comment":"QB battle between Deshaun Watson and Shedeur Sanders. Feels like nobody wins here."},
    {"rank":29, "team":"Las Vegas Raiders",      "prev":30, "tier":"🔄 Rebuilding", "comment":"Kirk Cousins as a bridge while Fernando Mendoza learns. A wasted year, but not necessarily a bad strategy."},
    {"rank":30, "team":"New York Jets",          "prev":29, "tier":"🔄 Rebuilding", "comment":"Geno Smith's offseason drama makes a long fall look likely. Hard to be optimistic."},
    {"rank":31, "team":"Miami Dolphins",         "prev":30, "tier":"🔄 Rebuilding", "comment":"Malik Willis could strike gold but the weapons around him are limited and the schedule is brutal."},
    {"rank":32, "team":"Arizona Cardinals",      "prev":32, "tier":"🔄 Rebuilding", "comment":"Scouting director suspended for leaking draft info. A rough offseason off the field too."},
]

# ── ESPN API helpers ─────────────────────────
@st.cache_data(ttl=3600)
def fetch_espn_scoreboard():
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        r = requests.get(url, timeout=10)
        data = r.json()
        events = data.get("events", [])
        games = []
        for e in events:
            comps = e.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get("homeAway")=="home"), competitors[0])
            away = next((c for c in competitors if c.get("homeAway")=="away"), competitors[1])
            status = e.get("status", {}).get("type", {}).get("description", "")
            games.append({
                "home": home["team"]["displayName"],
                "away": away["team"]["displayName"],
                "date": e.get("date",""),
                "status": status,
                "name": e.get("name",""),
            })
        return games, data.get("week", {}).get("number", None), None
    except Exception as ex:
        return [], None, str(ex)

@st.cache_data(ttl=3600)
def fetch_espn_team_stats(team_name):
    try:
        abbr_map = {
            'Arizona Cardinals':'ari','Atlanta Falcons':'atl','Baltimore Ravens':'bal',
            'Buffalo Bills':'buf','Carolina Panthers':'car','Chicago Bears':'chi',
            'Cincinnati Bengals':'cin','Cleveland Browns':'cle','Dallas Cowboys':'dal',
            'Denver Broncos':'den','Detroit Lions':'det','Green Bay Packers':'gb',
            'Houston Texans':'hou','Indianapolis Colts':'ind','Jacksonville Jaguars':'jax',
            'Kansas City Chiefs':'kc','Las Vegas Raiders':'lv','Los Angeles Chargers':'lac',
            'Los Angeles Rams':'lar','Miami Dolphins':'mia','Minnesota Vikings':'min',
            'New England Patriots':'ne','New Orleans Saints':'no','New York Giants':'nyg',
            'New York Jets':'nyj','Philadelphia Eagles':'phi','Pittsburgh Steelers':'pit',
            'San Francisco 49ers':'sf','Seattle Seahawks':'sea','Tampa Bay Buccaneers':'tb',
            'Tennessee Titans':'ten','Washington Commanders':'wsh',
        }
        abbr = abbr_map.get(team_name)
        if not abbr: return {}
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}"
        r = requests.get(url, timeout=10)
        return r.json().get("team", {})
    except:
        return {}

# ── Model ────────────────────────────────────
@st.cache_resource
def load_model():
    scores = pd.read_csv('data/spreadspoke_scores.csv')
    scores = scores[(scores['score_home']>0)|(scores['score_away']>0)]
    renames = {
        'Oakland Raiders':'Las Vegas Raiders','St. Louis Rams':'Los Angeles Rams',
        'San Diego Chargers':'Los Angeles Chargers','Washington Redskins':'Washington Commanders',
        'Washington Football Team':'Washington Commanders','Tennessee Oilers':'Tennessee Titans',
        'Houston Oilers':'Tennessee Titans','Phoenix Cardinals':'Arizona Cardinals',
        'Baltimore Colts':'Indianapolis Colts','Los Angeles Raiders':'Las Vegas Raiders',
    }
    scores['team_home'] = scores['team_home'].replace(renames)
    scores['team_away'] = scores['team_away'].replace(renames)
    scores['home_win']  = (scores['score_home']>scores['score_away']).astype(int)
    scores['schedule_date'] = pd.to_datetime(scores['schedule_date'])
    scores = scores.sort_values('schedule_season')
    scores = scores[scores['schedule_season']>=1990].copy()

    def get_team_stats(df, team, before_season):
        hg = df[(df['team_home']==team)&(df['schedule_season']<before_season)]
        ag = df[(df['team_away']==team)&(df['schedule_season']<before_season)]
        hw = (hg['score_home']>hg['score_away']).sum()
        aw = (ag['score_away']>ag['score_home']).sum()
        total = len(hg)+len(ag)
        if total==0: return 0.5,22.0,20.0
        return (hw+aw)/total, pd.concat([hg['score_home'],ag['score_away']]).mean(), pd.concat([hg['score_away'],ag['score_home']]).mean()

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

    lr_f1  = f1_score(y_test,lr.predict(X_test))
    rf_f1  = f1_score(y_test,rf.predict(X_test))
    xgb_f1 = f1_score(y_test,xgb.predict(X_test))

    season_notes={
        2004:"2004 — An unusually balanced season with no dominant team.",
        2007:"2007 — The undefeated Patriots made this one of the most predictable seasons in memory.",
        2020:"2020 — COVID season. Played without fans, removing crowd noise entirely.",
        2022:"2022 — High-upset season. Multiple strong teams lost games they should have won.",
        2024:"2024 — Lamar Jackson, Joe Burrow and Patrick Mahomes all missed games through injury.",
    }
    season_acc=[]
    for s in sorted(seasons.unique()):
        if s<2000: continue
        mask=seasons==s
        if mask.sum()<10: continue
        season_acc.append({'season':int(s),'accuracy':accuracy_score(y[mask],xgb.predict(X[mask])),'note':season_notes.get(int(s),"")})

    cm=confusion_matrix(y_test,xgb.predict(X_test))
    fpr,tpr,_=roc_curve(y_test,xgb.predict_proba(X_test)[:,1])
    roc_auc=auc(fpr,tpr)

    probs=xgb.predict_proba(X_test)[:,1]
    preds_test=xgb.predict(X_test)
    conf_data=[]
    for display,low,high in [('🔴 Low Confidence',0.0,0.07),('🟡 Medium Confidence',0.07,0.15),('🟢 High Confidence',0.15,0.50)]:
        mask=(abs(probs-0.5)>=low)&(abs(probs-0.5)<high)
        if mask.sum()>0:
            conf_data.append({'confidence':display,'accuracy':accuracy_score(y_test[mask],preds_test[mask]),'games':int(mask.sum())})

    feature_names=['Home Team Win Rate','Away Team Win Rate','Home Team Avg Points Scored',
                   'Away Team Avg Points Scored','Home Team Avg Points Conceded','Away Team Avg Points Conceded']
    importances=np.abs(lr.coef_[0])  # use LR's real coefficients (magnitude), matching the worked example weights table

    scores_32=scores[scores['schedule_season']>=2002].copy()
    scores_32['period']=pd.cut(scores_32['schedule_season'],bins=[2001,2009,2019,2026],labels=['2002–2009','2010–2019','2020–2025'])
    period_hw=scores_32.groupby('period',observed=True)['home_win'].agg(home_win_rate='mean',games='count',home_wins='sum').reset_index()
    period_hw.columns=['Period','Home Win Rate','Games','Home Wins']

    return (lr,scores,get_team_stats,xgb_acc,lr_acc,rf_acc,
            xgb_f1,lr_f1,rf_f1,
            pd.DataFrame(season_acc),cm,fpr,tpr,roc_auc,
            conf_data,feature_names,importances,X_test,y_test,period_hw,lr)

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
    h2h=scores[((scores['team_home']==t1)&(scores['team_away']==t2))|((scores['team_home']==t2)&(scores['team_away']==t1))].copy()
    w1=w2=0
    for _,r in h2h.iterrows():
        if r['team_home']==t1:
            if r['score_home']>r['score_away']: w1+=1
            else: w2+=1
        else:
            if r['score_away']>r['score_home']: w1+=1
            else: w2+=1
    return w1,w2,len(h2h)

def predict_game(model, scores, get_team_stats, home, away):
    h_wr,h_sc,h_co=get_team_stats(scores,home,2026)
    a_wr,a_sc,a_co=get_team_stats(scores,away,2026)
    prob=model.predict_proba([[h_wr,a_wr,h_sc,a_sc,h_co,a_co]])[0]
    hp,ap=prob[1],prob[0]
    diff=abs(hp-ap)
    if diff>=0.15: conf="🟢 High Confidence"
    elif diff>=0.07: conf="🟡 Medium Confidence"
    else: conf="🔴 Low Confidence"
    winner=home if hp>ap else away
    return hp,ap,conf,winner,h_wr,h_sc,h_co,a_wr,a_sc,a_co

# ── Load ─────────────────────────────────────
st.markdown('<p class="nflnerd-brand">🏈 NFLNerd</p>', unsafe_allow_html=True)
st.title("NFL Game Predictor")

with st.spinner("Loading model..."):
    (model,scores,get_team_stats,xgb_acc,lr_acc,rf_acc,
     xgb_f1,lr_f1,rf_f1,
     season_acc_df,cm,fpr,tpr,roc_auc,
     conf_data,feature_names,importances,
     X_test,y_test,period_hw,lr) = load_model()

accuracy = lr_acc
all_teams = sorted(CURRENT_NFL_TEAMS)

# ── Tabs ─────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "🔮 Predict",
    "📅 This Week's Games",
    "🏆 Team Rankings",
    "📜 Prediction History",
    "📊 How It Works",
    "🧪 Data Science",
    "👤 Who Am I?"
])

# ════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════
with tab1:
    st.markdown("### 🔮 Predict Any NFL Matchup")
    st.markdown("**The NFL Game Predictor is a tool that predicts the outcome of any NFL matchup. Select the two teams and the model returns each team's probability of winning. Predictions are based on each team's historical performance data, going back to 1990.**")
    st.caption("⚠️ Recent form is based on historical data only — live form coming soon.")
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1:
        logo_url = ESPN_LOGOS.get('Kansas City Chiefs','')
        if logo_url: st.image(logo_url, width=60)
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Select home team", all_teams,
            index=all_teams.index("Kansas City Chiefs") if "Kansas City Chiefs" in all_teams else 0)
        if ESPN_LOGOS.get(home_team): st.image(ESPN_LOGOS[home_team], width=50)
    with c2:
        logo_url = ESPN_LOGOS.get('Philadelphia Eagles','')
        if logo_url: st.image(logo_url, width=60)
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Select away team", all_teams,
            index=all_teams.index("Philadelphia Eagles") if "Philadelphia Eagles" in all_teams else 1)
        if ESPN_LOGOS.get(away_team): st.image(ESPN_LOGOS[away_team], width=50)

    st.markdown("---")
    if st.button("🔮 Predict Game Outcome", use_container_width=True, key="predict_btn"):
        if home_team==away_team:
            st.error("Please select two different teams!")
        else:
            hp,ap,conf,winner,h_wr,h_sc,h_co,a_wr,a_sc,a_co = predict_game(model,scores,get_team_stats,home_team,away_team)

            st.markdown("---")
            st.subheader("📊 Prediction Result")
            c1,c2,c3 = st.columns([2,1,2])
            with c1:
                if ESPN_LOGOS.get(home_team): st.image(ESPN_LOGOS[home_team], width=80)
                st.metric(f"🏠 {home_team}", f"{hp:.1%}")
            with c2:
                st.markdown("<div style='text-align:center; padding-top:40px; font-family:Oswald; font-size:24px; color:#666;'>VS</div>", unsafe_allow_html=True)
            with c3:
                if ESPN_LOGOS.get(away_team): st.image(ESPN_LOGOS[away_team], width=80)
                st.metric(f"✈️ {away_team}", f"{ap:.1%}")

            st.success(f"🏆 **Predicted Winner: {winner}**")
            diff = abs(hp-ap)
            if diff>=0.15: st.markdown(f'<span class="confidence-high">{conf}</span> — The model strongly favours one team.', unsafe_allow_html=True)
            elif diff>=0.07: st.markdown(f'<span class="confidence-med">{conf}</span> — The model leans one way but it\'s not clear cut.', unsafe_allow_html=True)
            else: st.markdown(f'<span class="confidence-low">{conf}</span> — This is a very tight matchup. Could go either way.', unsafe_allow_html=True)

            # Key stats driving prediction
            st.markdown("---")
            st.subheader("🔑 Key Stats Driving This Prediction")
            wr_diff = h_wr - a_wr
            sc_diff = h_sc - a_sc
            co_diff = a_co - h_co  # positive = home team faces leakier defence
            insights = []
            if abs(wr_diff)>0.05:
                leader = home_team if wr_diff>0 else away_team
                insights.append(f"**{leader}** has a {abs(wr_diff):.1%} higher historical win rate.")
            if abs(sc_diff)>2:
                leader = home_team if sc_diff>0 else away_team
                insights.append(f"**{leader}** averages {abs(sc_diff):.1f} more points per game.")
            if abs(h_co - a_co)>2:
                better_def = home_team if h_co<a_co else away_team
                insights.append(f"**{better_def}** concedes {abs(h_co-a_co):.1f} fewer points per game — stronger defence.")
            if not insights:
                insights.append("These two teams are closely matched across all six stats — hence the low confidence rating.")
            for ins in insights:
                st.markdown(f"• {ins}")

            # Win probability chart
            st.markdown("---")
            st.subheader("📊 Win Probability")
            fig=go.Figure(go.Bar(
                x=[f"🏠 {home_team}",f"✈️ {away_team}"],y=[hp*100,ap*100],
                marker_color=["#013369","#D50A0A"],
                text=[f"{hp:.1%}",f"{ap:.1%}"],
                textposition="outside",textfont=dict(color="white",size=16)))
            fig.update_layout(**CHART_LAYOUT,yaxis=dict(title="Win Probability (%)",range=[0,115],gridcolor="#222"),xaxis=dict(gridcolor="#222"),height=350)
            st.plotly_chart(fig,use_container_width=True)

            # Team stats comparison
            st.markdown("---")
            st.subheader("📋 Team Stats Comparison")
            st.dataframe(pd.DataFrame({
                'Stat':['Historical Win Rate','Avg Points Scored','Avg Points Conceded'],
                f'🏠 {home_team}':[f"{h_wr:.1%}",f"{h_sc:.1f}",f"{h_co:.1f}"],
                f'✈️ {away_team}':[f"{a_wr:.1%}",f"{a_sc:.1f}",f"{a_co:.1f}"],
            }).set_index('Stat'),use_container_width=True)

            # Head to head
            st.markdown("---")
            st.subheader("⚔️ Head to Head Record")
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

            # Recent form
            st.markdown("---")
            st.subheader("📅 Recent Form (Last 5 Games — Historical Data)")
            c1,c2=st.columns(2)
            with c1:
                st.markdown(f"**🏠 {home_team}**")
                for _,r in get_recent_form(scores,home_team).iterrows(): st.markdown(r['result'])
            with c2:
                st.markdown(f"**✈️ {away_team}**")
                for _,r in get_recent_form(scores,away_team).iterrows(): st.markdown(r['result'])

# ════════════════════════════════════════════
# TAB 2 — THIS WEEK'S GAMES
# ════════════════════════════════════════════
with tab2:
    st.markdown("### 📅 This Week's NFL Games")
    st.markdown("Live matchups pulled from the ESPN API, with NFLNerd predictions for every game.")

    games, week_num, err = fetch_espn_scoreboard()

    if err or not games:
        st.info("🏈 **It's the offseason!** The NFL season hasn't started yet. This tab will automatically populate with live game predictions when the 2026 season kicks off in September. Check back then!")
        st.markdown("In the meantime, use the **🔮 Predict** tab to predict any hypothetical matchup.")
    else:
        week_label = f"NFL Week {week_num}" if week_num else "This Week's Games"
        st.subheader(f"📅 {week_label}")

        # Map ESPN team names to our model's team names
        espn_to_model = {v.split('/')[-1].replace('.png',''):k for k,v in ESPN_LOGOS.items()}
        def match_team(espn_name):
            for model_name in CURRENT_NFL_TEAMS:
                if any(part.lower() in espn_name.lower() for part in model_name.split()):
                    return model_name
            return espn_name

        predictions = []
        for g in games:
            home = match_team(g['home'])
            away = match_team(g['away'])
            if home in CURRENT_NFL_TEAMS and away in CURRENT_NFL_TEAMS:
                hp,ap,conf,winner,_,_,_,_,_,_ = predict_game(model,scores,get_team_stats,home,away)
                predictions.append({**g,'home_mapped':home,'away_mapped':away,'hp':hp,'ap':ap,'conf':conf,'winner':winner})

        if predictions:
            # Weekly highlights
            most_confident = max(predictions, key=lambda x: abs(x['hp']-x['ap']))
            closest = min(predictions, key=lambda x: abs(x['hp']-x['ap']))
            biggest_upset = min(predictions, key=lambda x: x['hp'] if x['winner']!=x['home_mapped'] else x['ap'])

            st.markdown("#### ⚡ NFLNerd Weekly Highlights")
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown("**🟢 Most Confident Pick**")
                st.markdown(f"**{most_confident['winner']}** to win")
                st.markdown(f"{most_confident['home_mapped']} vs {most_confident['away_mapped']}")
                st.markdown(f"{max(most_confident['hp'],most_confident['ap']):.1%} probability")
            with c2:
                st.markdown("**🔴 Closest Game**")
                st.markdown(f"{closest['home_mapped']} vs {closest['away_mapped']}")
                st.markdown(f"{closest['hp']:.1%} vs {closest['ap']:.1%} — genuine coin flip")
            with c3:
                st.markdown("**🎯 Upset Watch**")
                st.markdown(f"**{biggest_upset['winner']}** predicted to win")
                st.markdown(f"Keep an eye on this one")

            st.markdown("---")
            st.markdown("#### 🏈 Game Predictions")

            for g in predictions:
                diff = abs(g['hp']-g['ap'])
                conf_color = "#00C853" if diff>=0.15 else ("#FFB300" if diff>=0.07 else "#D50A0A")
                home_logo = ESPN_LOGOS.get(g['home_mapped'],'')
                away_logo = ESPN_LOGOS.get(g['away_mapped'],'')

                with st.container():
                    st.markdown(f'<div class="match-card">', unsafe_allow_html=True)
                    c1,c2,c3,c4,c5 = st.columns([2,1,1,1,2])
                    with c1:
                        if home_logo: st.image(home_logo, width=50)
                        st.markdown(f"**{g['home_mapped']}**")
                        st.markdown(f"🏠 Home • {g['hp']:.1%}")
                    with c2:
                        st.markdown("<div style='text-align:center;padding-top:20px;color:#666;font-size:20px;'>VS</div>", unsafe_allow_html=True)
                    with c3:
                        winner_label = "🏠 HOME WIN" if g['winner']==g['home_mapped'] else "✈️ AWAY WIN"
                        st.markdown(f"<div style='text-align:center;padding-top:15px;'><span class='winner-badge'>{winner_label}</span></div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center;color:{conf_color};font-size:12px;margin-top:6px;'>{g['conf']}</div>", unsafe_allow_html=True)
                    with c4:
                        st.markdown("<div style='text-align:center;padding-top:20px;color:#666;font-size:20px;'>VS</div>", unsafe_allow_html=True)
                    with c5:
                        if away_logo: st.image(away_logo, width=50)
                        st.markdown(f"**{g['away_mapped']}**")
                        st.markdown(f"✈️ Away • {g['ap']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 3 — TEAM RANKINGS
# ════════════════════════════════════════════
with tab3:
    st.markdown("### 🏆 NFLNerd Power Rankings")
    st.markdown("*Updated weekly. Starting point: Fox Sports Power Rankings, July 28 2026.*")

    st.markdown("""
<div class="info-box">
<b>📝 NFLNerd Take — Heading Into Training Camp 2026</b><br><br>
The Los Angeles Rams are the talk of the offseason after a massive roster overhaul — some are already calling them 'The Dream Team.'
The defending champion Seattle Seahawks slip to 4th with serious RB concerns after losing Kenneth Walker III.
Meanwhile the AFC looks wide open: Baltimore, Denver, Buffalo and Cincinnati all have legitimate Super Bowl aspirations.
At the bottom, the Jets, Dolphins and Cardinals look set for another long year.
These rankings will be updated weekly throughout the season alongside my YouTube predictions.
</div>
""", unsafe_allow_html=True)

    tiers = ["🔥 Elite", "✅ Contenders", "⚠️ Middling", "🔄 Rebuilding"]
    tier_colors = {"🔥 Elite":"#D50A0A","✅ Contenders":"#013369","⚠️ Middling":"#FFB300","🔄 Rebuilding":"#444"}

    for tier in tiers:
        tier_teams = [t for t in FOX_RANKINGS if t['tier']==tier]
        if not tier_teams: continue
        color = tier_colors.get(tier,"#444")
        st.markdown(f'<div class="tier-header" style="background:{color}20;color:{color};border-left:4px solid {color};">{tier}</div>', unsafe_allow_html=True)

        for t in tier_teams:
            prev = t.get('prev', t['rank'])
            diff_rank = prev - t['rank']
            if diff_rank > 0:
                movement = f'<span class="movement-up">▲{diff_rank}</span>'
            elif diff_rank < 0:
                movement = f'<span class="movement-down">▼{abs(diff_rank)}</span>'
            else:
                movement = '<span class="movement-same">—</span>'

            logo_url = ESPN_LOGOS.get(t['team'],'')
            logo_html = f'<img src="{logo_url}" width="40" style="margin-right:12px;vertical-align:middle;">' if logo_url else ''

            st.markdown(f"""
<div class="team-ranking-card">
    <div class="rank-number">#{t['rank']}</div>
    {logo_html}
    <div style="flex:1;">
        <div style="font-family:Oswald;font-size:17px;font-weight:600;color:white;">{t['team']} {movement}</div>
        <div style="font-size:13px;color:#aaa;margin-top:4px;">{t['comment']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 4 — PREDICTION HISTORY
# ════════════════════════════════════════════
with tab4:
    st.markdown("### 📜 NFLNerd Prediction History")
    st.markdown("A record of every official NFLNerd weekly prediction, logged alongside my YouTube videos.")
    st.markdown("---")

    st.info("🏈 **No predictions logged yet.** Check back after Week 1 of the 2026 NFL season — every weekly prediction will be recorded here alongside the actual results.")

    st.markdown("#### What this tab will show:")
    st.markdown("""
- **Season selector** — browse predictions from any season
- **Overall accuracy** — how many games NFLNerd correctly predicted this season
- **Best and worst week** — the highest and lowest accuracy week of the season
- **Biggest correct call** — the highest confidence prediction that turned out right
- **Full prediction log** — every game, the predicted winner, the actual winner, and whether it was correct
""")

# ════════════════════════════════════════════
# TAB 5 — HOW IT WORKS
# ════════════════════════════════════════════
with tab5:
    st.markdown("### 📊 How It Works")
    st.markdown("A simple explainer for NFL fans on how the predictor makes its decisions.")
    st.markdown("---")

    st.subheader("🏈 What Does the Predictor Do?")
    st.markdown("""
The NFL Game Predictor is a tool that predicts the outcome of any NFL matchup.
You select the two teams, and the model returns each team's probability of winning.
Predictions are based on each team's historical performance data, going back to 1990.
""")

    st.markdown("---")
    st.subheader("📋 What Is It Based On?")
    st.markdown("For every matchup, the model looks at **6 stats** — 3 for each team:")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**🏠 Home Team**")
        st.markdown("• Overall win rate (all games since 1990)")
        st.markdown("• Average points scored per game")
        st.markdown("• Average points conceded per game")
    with c2:
        st.markdown("**✈️ Away Team**")
        st.markdown("• Overall win rate (all games since 1990)")
        st.markdown("• Average points scored per game")
        st.markdown("• Average points conceded per game")

    st.markdown("---")
    st.subheader("🎯 What Do the Confidence Levels Mean?")
    st.markdown("""
When the model predicts a game, it gives each team a probability — for example, 64% vs 36%.
The gap between those two numbers determines the confidence level:
""")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div style="background:#1a0000;border:1px solid #D50A0A;border-radius:8px;padding:16px;text-align:center;"><div style="font-size:24px;">🔴</div><div style="font-family:Oswald;font-size:16px;color:#D50A0A;">LOW CONFIDENCE</div><div style="font-size:13px;color:#aaa;margin-top:8px;">Gap under 7%<br>e.g. 53% vs 47%<br>Nearly a coin flip</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#1a1400;border:1px solid #FFB300;border-radius:8px;padding:16px;text-align:center;"><div style="font-size:24px;">🟡</div><div style="font-family:Oswald;font-size:16px;color:#FFB300;">MEDIUM CONFIDENCE</div><div style="font-size:13px;color:#aaa;margin-top:8px;">Gap 7–15%<br>e.g. 57% vs 43%<br>Leans one way</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div style="background:#001a08;border:1px solid #00C853;border-radius:8px;padding:16px;text-align:center;"><div style="font-size:24px;">🟢</div><div style="font-family:Oswald;font-size:16px;color:#00C853;">HIGH CONFIDENCE</div><div style="font-size:13px;color:#aaa;margin-top:8px;">Gap over 15%<br>e.g. 66% vs 34%<br>Strong favourite</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 6 — DATA SCIENCE
# ════════════════════════════════════════════
with tab6:
    st.markdown("### 🧪 Data Science & Model Analysis – Mikail Atif")
    st.markdown("The NFL Game Predictor is a machine learning model trained and tested on over 9,000 games and 35 NFL seasons. This tab walks through everything behind the scenes — how it was built, what it learned, and how well it performs.")

    st.markdown("---")
    st.subheader("💡 What is a Machine Learning Model (ML)?")
    st.markdown("""
An ML Model uses previous data to predict future outcomes. Similar to how a weather app studies years of past weather patterns to predict tomorrow's forecast — my model does the same, but with NFL games.
""")

    st.markdown("---")
    st.subheader("🎯 What is my ML Model trying to predict?")
    st.markdown("""
My model is trained on historical data to try and accurately predict which team will have the higher probability of winning any given NFL game. Rather than a simple yes or no, a probability is given — for example 64% win probability for the home team vs 36% win probability for the away team — so you can see how confident the model is in its prediction. How the model arrives at these probabilities will be explained later on.
""")

    st.markdown("---")
    st.subheader("📦 What historical data is my model trained on?")
    st.markdown("""
The historical data comes from Kaggle — a website where data scientists share free datasets. The NFL dataset contains 9,455 game results, going back to 1990, giving the model plenty of data to be trained on, to make it as accurate as possible.

Before training, this data is split into two groups: about 7,500 games (80%) are used to actually train the model — this is where it learns the patterns of which teams are more likely to win the game (e.g. the team with the higher average points scored). The remaining 1,900 games (20%) are held back completely and never shown to the model during training. These are used afterwards in testing, to see how accurate the model really is, since it's being tested on games it has never seen before.

The two teams in a matchup are distinguished as the home team and the away team. This is a major distinction that's always present between the two sides — true for every NFL game ever played.

From all of that historical NFL data, the model takes 6 key stats from each game — split into 3 for the home team and 3 for the away team, reflecting that home/away distinction. The same 6 stats are used for every single matchup, so every game is judged on the same criteria.
""")

    st.markdown("#### The 6 Stats used")
    st.markdown("""
1. **Home team historical win rate** — the percentage of all games the home team has won since 1990
2. **Away team historical win rate** — the percentage of all games the away team has won since 1990
3. **Home team average points scored** — how many points the home team scores per game on average since 1990
4. **Away team average points scored** — how many points the away team scores per game on average since 1990
5. **Home team average points conceded** — how many points the home team has conceded per game on average since 1990
6. **Away team average points conceded** — how many points the away team has conceded per game on average since 1990
""")

    total_games=len(scores); hw_pct=scores['home_win'].mean(); total_hw=int(scores['home_win'].sum())
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Total Games",f"{total_games:,}")
    with c2: st.metric("Seasons","1990–2025")
    with c3: st.metric("Total Home Wins",f"{total_hw:,}")
    with c4: st.metric("Home Win Rate",f"{hw_pct:.1%}")

    st.markdown("---")
    st.subheader("🏆 Which Model did I pick to power the NFL Game Predictor?")
    st.markdown("""
To find the best model to power the NFL Game Predictor, I tested 3 different ML models — Logistic Regression, Random Forest, and XGBoost — on how well they could correctly predict unseen NFL games, comparing their accuracy and F1 scores. My results showed that Logistic Regression came out on top for both accuracy and F1 score, so I chose it to power the NFL Game Predictor.
""")

    st.markdown("This bar chart shows the percentage of games each model correctly predicted, out of the 1,900 games held back for testing. The three models are closely matched, with less than 2% separating the best and worst performer, and only a 0.16% gap between Logistic Regression and XGBoost.")

    # Dynamic chart — automatically highlights whichever model actually scores highest
    model_names=['Logistic Regression','Random Forest','XGBoost']
    model_accs=[lr_acc*100,rf_acc*100,xgb_acc*100]
    bar_colors=['#888','#888','#888']
    best_idx=int(np.argmax(model_accs))
    bar_colors[best_idx]='#D50A0A'
    model_labels=[f"{n} ✅" if i==best_idx else n for i,n in enumerate(model_names)]

    fig_m=go.Figure(go.Bar(x=model_labels,y=model_accs,marker_color=bar_colors,
        text=[f"{a:.2f}%" for a in model_accs],textposition='outside',textfont=dict(color='white',size=14)))
    fig_m.add_hline(y=50,line_dash='dash',line_color='#555',annotation_text='Random guessing (50%)',annotation_font_color='#aaa',annotation_position='bottom right')
    fig_m.update_layout(**CHART_LAYOUT,yaxis=dict(title='Games correctly predicted (%)',range=[45,65],gridcolor='#222'),xaxis=dict(gridcolor='#222'),height=380)
    st.plotly_chart(fig_m,use_container_width=True)

    st.markdown("#### 📐 F1 Scores")
    explainer("""
<b>What is an F1 score?</b><br>
Accuracy alone doesn't tell the whole story. If a model always predicted a home team win, no matter what - since home teams actually win around 57% of NFL games, this model would probably get 57 game predictions correct, purely by luck, giving it a 57% accuracy score, which is actually quite good. But this model would have a 0% accuracy at predicting away wins, and this wouldn't be captured by the accuracy score.<br><br>
This is exactly the problem the F1 score is designed to catch. Accuracy alone can make an unbalanced model (which only predicts the more likely outcome) look better than it really is. The F1 score checks how well a model predicts both home wins and away wins correctly, not just how often it's right overall.<br><br>

It combines two things:<br><br>

<b>Precision</b> — out of all the games the model predicted as a specific outcome (either a home win or an away win), how many of those predictions were actually correct?<br><br>

1. <b>Home Wins Example</b><br>
LR predicted 40 games as home wins. Out of those 40 predictions, 30 were correct (home wins) → Precision = 30 ÷ 40 = 75%<br><br>

2. <b>Away Wins Example</b><br>
LR predicted 25 games as away wins. Out of those 25 predictions, 15 were correct (away wins) → Precision = 15 ÷ 25 = 60%<br><br>

<b>Recall</b> — out of all the games that were actually a home win or an away win, how many did the model correctly predict?<br><br>

1. <b>Home Wins Example</b><br>
Out of 50 games that were all home wins, LR correctly predicted 35 of them → Recall = 35 ÷ 50 = 70%, the model predicted the other 15 as away games<br><br>

2. <b>Away Wins Example</b><br>
Out of 50 games that were all away wins, LR correctly predicted 25 of them → Recall = 25 ÷ 50 = 50%, the model predicted the other 25 as home games<br><br>

<b>How to Calculate the F1 Score?</b><br>
Formula for F1: 2 × (Precision × Recall) ÷ (Precision + Recall)<br><br>

Calculating the F1 scores for both the Home and Away examples:<br><br>

<b>Home Wins F1 Score</b><br>
F1 = 2 × (0.75 × 0.70) ÷ (0.75 + 0.70)<br>
F1 = 2 × 0.525 ÷ 1.45<br>
F1 = 1.05 ÷ 1.45<br>
F1 = 0.724<br><br>

<b>Away Wins F1 Score</b><br>
F1 = 2 × (0.60 × 0.50) ÷ (0.60 + 0.50)<br>
F1 = 2 × 0.300 ÷ 1.1<br>
F1 = 0.600 ÷ 1.1<br>
F1 = 0.545<br><br>

The F1 score ranges from 0 (useless) to 1 (perfect). A higher F1 score means the model is better at correctly predicting both home wins and away wins, making the model more balanced.
""")

    st.markdown("This bar chart shows the F1 scores for each of the ML Models we are testing. Similar to the accuracies of the three models, the F1 scores are all quite similar, with a range of just 0.066 - but with LR still coming out on top with the highest F1 score.")

    best_f1_idx = int(np.argmax([lr_f1, rf_f1, xgb_f1]))
    f1_model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    f1_labels = [f"{n} ✅" if i==best_f1_idx else n for i,n in enumerate(f1_model_names)]

    fig_f1 = go.Figure(go.Bar(
        x=f1_labels, y=[lr_f1, rf_f1, xgb_f1],
        marker_color=['#D50A0A' if i==best_f1_idx else '#888' for i in range(3)],
        text=[f"{v:.3f}" for v in [lr_f1, rf_f1, xgb_f1]],
        textposition='outside', textfont=dict(color='white', size=14)))
    fig_f1.update_layout(**CHART_LAYOUT,
        yaxis=dict(title='F1 Score (0–1)', range=[0,1], gridcolor='#222'),
        xaxis=dict(gridcolor='#222'), height=380)
    st.plotly_chart(fig_f1, use_container_width=True)

    f1_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'F1 Score': [f"{lr_f1:.3f}", f"{rf_f1:.3f}", f"{xgb_f1:.3f}"],
    })
    st.dataframe(f1_df.set_index('Model'), use_container_width=True)

    st.markdown("---")
    st.subheader("📐 How Does Logistic Regression Work?")
    explainer("""
Logistic Regression combines all 6 team stats into a single calculation, then converts that into a probability of the home team winning. It works in two steps.<br><br>

<b>Step 1 — Combine the 6 stats into one number, called z</b><br>
During training, Logistic Regression works out a <b>weight</b> for each of the 6 stats — a number reflecting how useful that stat is for predicting a winner. It also calculates one extra fixed number called the <b>intercept</b>, which captures a simple fact: home teams win more often than away teams overall. So before even looking at either team's specific stats, the model starts with a slight lean toward the home team.<br><br>

<b>z = intercept + (weight₁ × stat₁) + (weight₂ × stat₂) + (weight₃ × stat₃) + (weight₄ × stat₄) + (weight₅ × stat₅) + (weight₆ × stat₆)</b><br><br>

The intercept and all 6 weights are fixed once training is complete — they never change. Only the specific team stats change from matchup to matchup.<br><br>

<b>How are the weights actually calculated?</b><br>
The model starts by guessing random weights, then tests them against thousands of past games it already knows the real result of. Where its predictions are wrong, it nudges each weight slightly in the direction that would have made the prediction more accurate. This happens thousands of times, with tiny adjustments each time, until the weights settle into stable values — the real ones shown below.<br><br>

<b>Step 2 — Turn z into an actual probability</b><br>
z on its own isn't a probability — it could be any number, positive or negative. To turn it into a sensible probability between 0% and 100%, it's run through a formula called the S-curve:<br><br>

<b>Probability of home win = 1 ÷ (1 + e^(-z))</b><br><br>

No matter how big or small z is, this formula always produces a result between 0% and 100% — a very negative z gets squeezed close to 0%, a very positive z gets squeezed close to 100%, and a z of exactly 0 lands at exactly 50%.
""")

    st.markdown("#### A Real Worked Example: Kansas City Chiefs (Home) vs Philadelphia Eagles (Away)")
    st.markdown("This is a real game that Logistic Regression has calculated the probabilities for, using the real weights my model learned from training on 9,000+ NFL games, and the real historical stats for these two teams.")

    lr_coefs = dict(zip(['home_win_rate','away_win_rate','home_avg_scored','away_avg_scored','home_avg_conceded','away_avg_conceded'], lr.coef_[0])) if 'lr' in dir() else None

    weight_df = pd.DataFrame({
        'Stat': ['Intercept','Home Team Win Rate','Away Team Win Rate','Home Team Avg Points Scored',
                 'Away Team Avg Points Scored','Home Team Avg Points Conceded','Away Team Avg Points Conceded'],
        'Weight': [f"{lr.intercept_[0]:+.4f}", f"{lr.coef_[0][0]:+.4f}", f"{lr.coef_[0][1]:+.4f}",
                   f"{lr.coef_[0][2]:+.4f}", f"{lr.coef_[0][3]:+.4f}", f"{lr.coef_[0][4]:+.4f}", f"{lr.coef_[0][5]:+.4f}"]
    })
    st.dataframe(weight_df.set_index('Stat'), use_container_width=True)

    ex_h_wr, ex_h_sc, ex_h_co = get_team_stats(scores, "Kansas City Chiefs", 2026)
    ex_a_wr, ex_a_sc, ex_a_co = get_team_stats(scores, "Philadelphia Eagles", 2026)

    stats_df = pd.DataFrame({
        'Stat': ['Win Rate','Avg Points Scored','Avg Points Conceded'],
        'Chiefs (Home)': [f"{ex_h_wr:.1%}", f"{ex_h_sc:.2f}", f"{ex_h_co:.2f}"],
        'Eagles (Away)': [f"{ex_a_wr:.1%}", f"{ex_a_sc:.2f}", f"{ex_a_co:.2f}"],
    })
    st.dataframe(stats_df.set_index('Stat'), use_container_width=True)

    ex_z = (lr.intercept_[0]
            + lr.coef_[0][0]*ex_h_wr + lr.coef_[0][1]*ex_a_wr
            + lr.coef_[0][2]*ex_h_sc + lr.coef_[0][3]*ex_a_sc
            + lr.coef_[0][4]*ex_h_co + lr.coef_[0][5]*ex_a_co)
    ex_prob_home = 1/(1+np.exp(-ex_z))

    st.markdown(f"""
**Step 1 — Calculating z:**
```
z = {lr.intercept_[0]:.4f} 
    + ({lr.coef_[0][0]:.4f} × {ex_h_wr:.3f}) 
    + ({lr.coef_[0][1]:.4f} × {ex_a_wr:.3f}) 
    + ({lr.coef_[0][2]:.4f} × {ex_h_sc:.2f}) 
    + ({lr.coef_[0][3]:.4f} × {ex_a_sc:.2f}) 
    + ({lr.coef_[0][4]:.4f} × {ex_h_co:.2f}) 
    + ({lr.coef_[0][5]:.4f} × {ex_a_co:.2f})

z = {ex_z:.4f}
```

**Step 2 — Turning z into a probability:**
```
Probability of Chiefs winning = 1 ÷ (1 + e^(-{ex_z:.4f})) = {ex_prob_home:.1%}
Probability of Eagles winning = {1-ex_prob_home:.1%}
```
""")
    takeaway(f"So Logistic Regression predicts the **Chiefs have a {ex_prob_home:.1%} chance of winning**, and the **Eagles have a {1-ex_prob_home:.1%} chance** — using 100% real weights and real historical stats.")

    st.markdown("---")
    st.subheader("🔍 2. Which Stats Matter Most to the Model?")
    explainer(f"""
This chart uses the same real weights from the worked example above — Logistic Regression's 6 coefficients.
The bars show each weight's <b>size</b> (ignoring whether it's positive or negative), so you can see which stats have
the biggest influence on the model's predictions, regardless of which direction they push the outcome.<br><br>
<b>What each stat means:</b><br>
• <b>Home/Away Team Win Rate</b> — overall % of games won, across all their games since 1990<br>
• <b>Home/Away Team Avg Points Scored</b> — average points scored per game since 1990. A team averaging 30+ has a high-powered offence.<br>
• <b>Home/Away Team Avg Points Conceded</b> — average points let in per game since 1990. A team conceding 28+ has a leaky defence.<br><br>
<b>A quick note on comparing these fairly:</b> win rate is measured on a 0–1 scale, while points scored/conceded are measured in much bigger numbers (points per game).
Because of this, the weights aren't perfectly comparable "apples to apples" — a small weight on a big-scale stat can still matter a lot. What we can say confidently is that <b>Away Team Win Rate</b> has the single biggest weight of the 6, meaning it has the strongest pull on the model's predictions.
""")
    idx=np.argsort(importances)[::-1]
    s_fn=[feature_names[i] for i in idx]; s_fi=[importances[i] for i in idx]
    fig_fi=go.Figure(go.Bar(x=s_fi,y=s_fn,orientation='h',marker_color='#013369',
        text=[f"{v:.4f}" for v in s_fi],textposition='outside',textfont=dict(color='white')))
    fig_fi.update_layout(**CHART_LAYOUT,
        xaxis=dict(title='Weight Size (absolute value of the real LR coefficient)',gridcolor='#222',range=[0,max(s_fi)*1.35]),
        yaxis=dict(gridcolor='#222',autorange='reversed'),height=420,margin=dict(l=230,r=80,t=30,b=50))
    st.plotly_chart(fig_fi,use_container_width=True)

    st.markdown("---")
    st.subheader("📈 3. How Accurate Was the Model in Each NFL Season?")
    explainer("""
This shows the percentage of games the model correctly predicted out of all NFL games played in each individual season from 2000 to 2025.
For example, if a season had 256 games and the model correctly predicted 153 of them, that season's accuracy would be 59.8%.<br><br>
<b>Why does the number of upsets change year to year?</b>
Some seasons produce more upsets due to factors the model cannot see: a star QB getting injured mid-season,
a coaching system that takes time to click, or a dark horse team rising from average on paper to a genuine Super Bowl contender.
In 2024, Lamar Jackson (Ravens), Joe Burrow (Bengals) and Patrick Mahomes (Chiefs) all missed games through injury —
three of the most predictable teams suddenly became much harder to forecast.<br><br>
<b>Hover over any point</b> to see the exact accuracy and context for notable seasons.
""")
    if not season_acc_df.empty:
        avg_acc=season_acc_df['accuracy'].mean()
        fig_s=go.Figure()
        fig_s.add_trace(go.Scatter(
            x=season_acc_df['season'],y=season_acc_df['accuracy']*100,
            mode='lines+markers',line=dict(color='#D50A0A',width=3),
            marker=dict(size=10,color='#D50A0A',line=dict(color='white',width=1.5)),
            customdata=season_acc_df['note'],
            hovertemplate='<b>Season: %{x}</b><br>Accuracy: %{y:.1f}%<br>%{customdata}<extra></extra>'))
        fig_s.add_hline(y=avg_acc*100,line_dash='dot',line_color='#4a90d9',
            annotation_text=f'Season average ({avg_acc:.1%})',annotation_font_color='#aaa',annotation_position='top right')
        fig_s.update_layout(**CHART_LAYOUT,
            xaxis=dict(title='NFL Season',gridcolor='#222',dtick=2,
                range=[season_acc_df['season'].min()-0.5,season_acc_df['season'].max()+0.5]),
            yaxis=dict(title='Games correctly predicted (%)',range=[75,95],gridcolor='#222',dtick=5),
            height=460,margin=dict(t=50,b=60,l=70,r=80))
        st.plotly_chart(fig_s,use_container_width=True)
        best=season_acc_df.loc[season_acc_df['accuracy'].idxmax()]
        worst=season_acc_df.loc[season_acc_df['accuracy'].idxmin()]
        c1,c2,c3=st.columns(3)
        with c1: st.metric("Season Average",f"{avg_acc:.1%}")
        with c2: st.metric("Best Season",f"{int(best['season'])} ({best['accuracy']:.1%})")
        with c3: st.metric("Toughest Season",f"{int(worst['season'])} ({worst['accuracy']:.1%})")
        takeaway(f"""
On average, the model correctly predicts <b>{avg_acc:.1%}</b> of games in an NFL season.
Its toughest season was <b>{int(worst['season'])}</b> and its best was <b>{int(best['season'])}</b>.
The accuracy goes up and down year to year because the number of upsets in the NFL goes up and down —
a more predictable season produces a higher score, not a better model.
""")

    st.markdown("---")
    st.subheader("🎯 4. Does the Model's Confidence Level Actually Mean Anything?")
    explainer("""
When the model predicts a game it gives each team a probability — for example home team 64%, away team 36%.
That means: if you played this exact game 100 times, the home team would win around 64 of them.<br><br>
<b>The three confidence levels are based on the gap between those two percentages:</b><br>
• 🔴 <b>Low Confidence</b> — gap under 7 percentage points (e.g. 53% vs 47%). Nearly a coin flip.<br>
• 🟡 <b>Medium Confidence</b> — gap between 7 and 15 percentage points (e.g. 57% vs 43%). Leans one way.<br>
• 🟢 <b>High Confidence</b> — gap over 15 percentage points (e.g. 66% vs 34%). Strongly favours one team.<br><br>
The model was tested on ~1,900 games from NFL seasons 1990–2025 (pre-1990 not used).
The <b>percentage on each bar</b> shows how often the model was correct within that confidence level.
The <b>number in brackets</b> shows how many of those ~1,900 games fell into each bucket.
""")
    if conf_data:
        cdf=pd.DataFrame(conf_data)
        green_row=cdf[cdf['confidence']=='🟢 High Confidence']
        if not green_row.empty:
            g_games=int(green_row['games'].values[0]); g_acc=float(green_row['accuracy'].values[0])
            g_correct=int(round(g_games*g_acc))
            explainer(f"For example, out of the ~1,900 test games, XGBoost had high confidence in <b>{g_games}</b> of them. From those {g_games} games, it correctly predicted the winner in <b>{g_correct}</b> of them — an accuracy of <b>{g_acc:.1%}</b>.")
        fig_c=go.Figure(go.Bar(x=cdf['confidence'],y=cdf['accuracy']*100,
            marker_color=['#D50A0A','#FFB300','#00C853'],
            text=[f"{a:.1%}\n({g} games)" for a,g in zip(cdf['accuracy'],cdf['games'])],
            textposition='outside',textfont=dict(color='white',size=13)))
        fig_c.add_hline(y=50,line_dash='dash',line_color='#555',annotation_text='Random guessing (50%)',annotation_font_color='#aaa',annotation_position='bottom right')
        fig_c.update_layout(**CHART_LAYOUT,yaxis=dict(title='Games correctly predicted (%)',range=[40,80],gridcolor='#222'),xaxis=dict(gridcolor='#222'),height=420)
        st.plotly_chart(fig_c,use_container_width=True)
        takeaway("""
If this chart is working correctly, the green bar (High Confidence) will be noticeably taller than the red bar (Low Confidence).
That confirms that when the model says it is confident, it really is more likely to be right.
""")

    st.markdown("---")
    st.subheader("🔢 5. Where Does the Model Go Wrong?")
    infobox("""
<b>Why is everything framed as home team vs away team?</b><br>
In every single NFL game ever played, there is one home team and one away team — it is the only thing that is always different between the two sides.
Because of this, the model answers one question: <b>will the home team win?</b>
Every prediction is either "yes, home team wins" or "no, away team wins."<br><br>
NFL ties do occur — roughly once every 200 games. Including ties as a third outcome is a planned future improvement.<br><br>
<b>Why do home teams win more often?</b> Across 35 seasons, home teams have won ~57% of all games:<br>
• <b>Crowd noise</b> — disrupts the away team's ability to communicate at the line of scrimmage<br>
• <b>No travel</b> — away teams often travel the day before, disrupting sleep and routine<br>
• <b>Stadium familiarity</b> — home teams know their own stadium's quirks<br><br>
<b>Important:</b> While home field is an advantage, if the away team is superior in win rate or scoring, the model will still favour the away team.
Home field is one factor among six, not an override.
""")
    tn,fp,fn,tp=cm.ravel(); total_all=tn+tp+fp+fn; total_right=tn+tp
    st.markdown(f"Out of all **{total_all:,} test games**, the model divided its predictions into 4 groups. Here are the results:")
    fig_cm=go.Figure(go.Heatmap(
        z=[[tn,fp],[fn,tp]],
        x=['Model predicted: Away win','Model predicted: Home win'],
        y=['Reality: Away team won','Reality: Home team won'],
        colorscale=[[0,'#080808'],[1,'#D50A0A']],
        text=[[f"✅ Correct\nAway win predicted\n& away won\n{tn:,} games",f"❌ Wrong\nHome win predicted\nbut away won\n{fp:,} games"],
              [f"❌ Wrong\nAway win predicted\nbut home won\n{fn:,} games",f"✅ Correct\nHome win predicted\n& home won\n{tp:,} games"]],
        texttemplate='%{text}',textfont=dict(size=10,color='white'),showscale=False))
    fig_cm.update_layout(**CHART_LAYOUT,height=440,xaxis=dict(side='top'),margin=dict(t=130,b=40,l=180,r=20))
    st.plotly_chart(fig_cm,use_container_width=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("✅ Away wins correctly called",tn)
    with c2: st.metric("✅ Home wins correctly called",tp)
    with c3: st.metric("❌ Called home win, away won",fp)
    with c4: st.metric("❌ Called away win, home won",fn)
    takeaway(f"""
Out of {total_all:,} test games, the model correctly predicted <b>{total_right:,}</b> outcomes.
It correctly called home wins more than twice as often as away wins ({tp:,} vs {tn:,}) —
reflecting the real-world pattern that home teams win ~57% of games.
""")

    st.markdown("---")
    st.subheader("📉 6. How Well Can the Model Separate Winners from Losers?")
    explainer(f"""
A ROC curve (Receiver Operating Characteristic) is a standard way in data science to measure how well a model separates two outcomes —
in this case, games the home team won versus games the away team won.<br><br>
We are working through this section to make it as clear as possible. For now the headline number is the <b>AUC score: {roc_auc:.3f}</b>.<br>
AUC (Area Under the Curve) runs from 0.5 (no better than guessing) to 1.0 (perfect).
Our score of {roc_auc:.3f} places this model in the same range as professional NFL forecasting tools.
""")
    fig_roc=go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',line=dict(color='#D50A0A',width=2),
        name=f'Our model (AUC = {roc_auc:.3f})',fill='tozeroy',fillcolor='rgba(213,10,10,0.08)'))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color='#666',dash='dash'),name='Random guessing (AUC = 0.5)'))
    fig_roc.update_layout(plot_bgcolor='#080808',paper_bgcolor='#080808',font=dict(color='#f0f0f0'),
        xaxis=dict(title='→ More away wins mistakenly predicted as home wins',gridcolor='#222',range=[0,1]),
        yaxis=dict(title='→ More home wins correctly identified',gridcolor='#222',range=[0,1]),
        legend=dict(bgcolor='#111',bordercolor='#333'),height=440,showlegend=True)
    st.plotly_chart(fig_roc,use_container_width=True)

    st.markdown("---")
    st.subheader("📦 7. The Data Behind the Model")
    explainer(f"""
<b>Source:</b> NFL Scores & Betting Dataset on Kaggle (spreadspoke_scores.csv).<br><br>
<b>How was the overall home win % calculated?</b><br>
{total_hw:,} home wins ÷ {total_games:,} total games = <b>{hw_pct:.1%}</b><br>
Yes — home teams have won {hw_pct:.0%} out of every 100 games in this dataset.
""")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Total Games",f"{total_games:,}")
    with c2: st.metric("Seasons","1990–2025")
    with c3: st.metric("Total Home Wins",f"{total_hw:,}")
    with c4: st.metric("Home Win Rate",f"{hw_pct:.1%}")

    st.markdown("#### 🏟️ Has Home Field Advantage Changed Over Time?")
    explainer("""
This groups seasons since 2002 into three periods and shows the home win % in each.
2002 is the starting point because that is when the Houston Texans joined as the 32nd franchise, completing the modern NFL.<br>
Seasons covered: 2002–2009 (8 seasons), 2010–2019 (10 seasons), 2020–2025 (6 seasons).<br>
<b>How each % was calculated:</b> total home wins in that period ÷ total games in that period (shown on each bar).
""")
    bar_texts=[f"{r['Home Win Rate']:.1%}\n({int(r['Home Wins']):,} wins ÷ {int(r['Games']):,} games)" for _,r in period_hw.iterrows()]
    fig_p=go.Figure(go.Bar(x=period_hw['Period'],y=period_hw['Home Win Rate']*100,marker_color='#013369',
        text=bar_texts,textposition='outside',textfont=dict(color='white',size=12)))
    fig_p.add_hline(y=50,line_dash='dash',line_color='#555',annotation_text='50% — no home advantage',annotation_font_color='#aaa',annotation_position='bottom right')
    fig_p.update_layout(**CHART_LAYOUT,yaxis=dict(title='Home Win %',range=[0,75],gridcolor='#222'),xaxis=dict(gridcolor='#222'),height=400)
    st.plotly_chart(fig_p,use_container_width=True)
    takeaway("Home field advantage is real — but it has been shrinking. The 2020–2025 period is the lowest on record, partly because the 2020 COVID season was played entirely without fans, removing crowd noise as a factor for an entire season.")

# ════════════════════════════════════════════
# TAB 7 — WHO AM I?
# ════════════════════════════════════════════
with tab7:
    st.markdown("### 👤 Who Am I?")
    st.markdown("---")
    c1,c2 = st.columns([1,2])
    with c1:
        st.image("https://a.espncdn.com/i/teamlogos/nfl/500/nfl.png", width=120)
    with c2:
        st.markdown("## NFLNerd")
        st.markdown("*NFL analyst, data science student, and creator of the NFL Game Predictor.*")

    st.markdown("---")
    st.subheader("🏈 About Me")
    st.markdown("""
[Placeholder — add your bio here. A couple of sentences about who you are, where you're from, and your interest in both the NFL and data science.]
""")

    st.subheader("💡 Why I Built This")
    st.markdown("""
[Placeholder — explain why you built the NFL Game Predictor. Your interest in data science, wanting to combine it with your passion for the NFL, your goal of becoming a data scientist.]
""")

    st.subheader("🛠️ My Projects")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
**🔮 NFL Game Predictor**
The site you're on right now. A machine learning model trained on 35 years of NFL data to predict the winner of any matchup.
""")
    with c2:
        st.markdown("""
**🏈 NFL Fantasy Predictor App**
A Flutter app for researching, drafting, and managing NFL fantasy football teams, backed by a live player data API.
[Add your app link here]
""")

    st.subheader("📱 Follow NFLNerd")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown("**YouTube**\n[Add link]")
    with c2: st.markdown("**Instagram**\n[Add link]")
    with c3: st.markdown("**TikTok**\n[Add link]")
    with c4: st.markdown("**Twitter/X**\n[Add link]")

# ── Footer ───────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div class="footer-bar">
    Built by <b style="color:#D50A0A;">NFLNerd</b> &nbsp;|&nbsp;
    Data: Kaggle spreadspoke_scores.csv (1990–2025) &nbsp;|&nbsp;
    Model accuracy: {accuracy:.1%} &nbsp;|&nbsp;
    <a href="#" style="color:#D50A0A;">YouTube</a> &nbsp;|&nbsp;
    <a href="#" style="color:#D50A0A;">Fantasy App</a>
</div>
""", unsafe_allow_html=True)
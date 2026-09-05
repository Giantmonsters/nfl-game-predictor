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
import re
from datetime import datetime, timedelta

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

def _parse_stat_number(s):
    """Best-effort parse of the leading number out of a stat's display
    string (handles thousands separators and % signs), for COMPARISON
    purposes only — never used for the text actually shown to the user.
    Returns None if nothing parseable, so callers can skip coloring rather
    than guess."""
    if s is None:
        return None
    cleaned = str(s).replace(",", "").replace("%", "").strip()
    match = re.match(r"^-?\d+(\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

def _compare_stat_pair(home_display, away_display, home_num=None, away_num=None):
    """Given a stat's display text for each side (and optionally the real
    underlying numeric value for each, when the display text is a compound
    string like '22/34, 198 YDS' where parsing the display text itself
    wouldn't give the right number to compare), return (home_html,
    away_html) with the higher value in green, the lower in red, and the
    numeric difference shown next to the winner. Falls back to plain,
    uncolored text if either side's number can't be determined."""
    h = home_num if home_num is not None else _parse_stat_number(home_display)
    a = away_num if away_num is not None else _parse_stat_number(away_display)
    if h is None or a is None or h == a:
        return home_display, away_display
    diff = abs(h - a)
    diff_str = f"+{diff:.1f}" if diff % 1 else f"+{int(diff)}"
    if h > a:
        home_html = f'<span class="movement-up">{home_display}</span> <span class="movement-up">({diff_str})</span>'
        away_html = f'<span class="movement-down">{away_display}</span>'
    else:
        home_html = f'<span class="movement-down">{home_display}</span>'
        away_html = f'<span class="movement-up">{away_display}</span> <span class="movement-up">({diff_str})</span>'
    return home_html, away_html

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
    """Live NFL scoreboard for the current week. Confirmed via live
    debugging (2026-09) that ESPN's scoreboard endpoint returns correct
    matchups for Week 1 with these exact parameters (16/16 games matched
    the real schedule) — the earlier appearance of wrong teams was actually
    a separate bug in match_team()'s fuzzy name matching, not bad data from
    this endpoint, so there's no need to special-case or hardcode anything
    here."""
    try:
        # Week 1 games start Sun 2026-09-13; NFL weeks run Tue-Mon, so the
        # Tuesday two days before (2026-09-08) is used as the week-1 anchor.
        # seasontype=2 = regular season only.
        season_week1_start = datetime(2026, 9, 8)
        days_since = (datetime.now() - season_week1_start).days
        current_week = max(1, min(18, (days_since // 7) + 1))

        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {"seasontype": 2, "week": current_week, "dates": 2026}
        r = requests.get(url, params=params, timeout=10)
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
            status_type = e.get("status", {}).get("type", {})
            games.append({
                "home": home["team"]["displayName"],
                "away": away["team"]["displayName"],
                "date": e.get("date",""),
                "status": status_type.get("description", ""),
                # state: 'pre' (upcoming), 'in' (live), 'post' (final) —
                # used to decide whether to show a prediction or a real score
                "state": status_type.get("state", "pre"),
                "status_detail": status_type.get("shortDetail", status_type.get("detail", "")),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "name": e.get("name",""),
            })
        return games, data.get("week", {}).get("number", current_week), None
    except Exception as ex:
        return [], None, str(ex)

# ── Shared ESPN team abbreviation map ────────
ESPN_ABBR = {
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

@st.cache_data(ttl=3600)
def fetch_espn_team_stats(team_name):
    try:
        abbr = ESPN_ABBR.get(team_name)
        if not abbr: return {}
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}"
        r = requests.get(url, timeout=10)
        return r.json().get("team", {})
    except:
        return {}

@st.cache_data(ttl=3600)
def fetch_espn_team_record(team_name, debug=False):
    """Team's current W-L record for the 2026 season (e.g. '0-0' before Week 1
    kicks off), pulled from the teams/{abbr} endpoint.

    This was removed once already after appearing to show real, plausible-
    looking-but-wrong records (e.g. '3-0') before Week 1 had even been
    played. That's the same root-cause bug pattern already fixed elsewhere
    in this app (Recent Form, Team Season Stats, Key Players): without an
    explicit season param, ESPN's "current" endpoints default to the most
    recent COMPLETED season's final data rather than the new season's
    (correctly empty) one. This explicitly passes season=2026 to force the
    real current season instead of silently falling back to 2025's final
    record. Returns None on any failure, so the UI can omit the record
    rather than show something wrong."""
    try:
        abbr = ESPN_ABBR.get(team_name)
        if not abbr:
            return None
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}"
        r = requests.get(url, params={"season": 2026}, timeout=10)
        team = r.json().get("team", {})
        if debug:
            st.write(f"DEBUG (team record) — record field for {team_name} with season=2026:")
            st.json(team.get("record", {}))
        record_items = team.get("record", {}).get("items", [])
        total = next((r for r in record_items if r.get("type") == "total"), None)
        return total.get("summary") if total else None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_espn_recent_form(team_name, n=5):
    """Live 'last N completed games' for a team, pulled from ESPN's team schedule
    endpoint. Returns None on any failure so callers can fall back to the
    static Kaggle-data version instead of showing nothing."""
    try:
        abbr = ESPN_ABBR.get(team_name)
        if not abbr:
            return None

        def get_events(season=None):
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}/schedule"
            # seasontype=2 = regular season only. Without this, ESPN's default
            # "current season" schedule includes preseason games too — since
            # the 2026 preseason has already finished but the 2026 regular
            # season hasn't started yet, that meant "most recent completed
            # game" was pulling preseason backups instead of real games.
            params = {"seasontype": 2}
            if season:
                params["season"] = season
            r = requests.get(url, params=params, timeout=10)
            return r.json().get("events", [])

        def completed_events(events):
            out = []
            for e in events:
                comp = e.get("competitions", [{}])[0]
                if comp.get("status", {}).get("type", {}).get("completed"):
                    out.append(e)
            return out

        events = completed_events(get_events())
        if not events:
            # Offseason fallback: current-season schedule has no completed
            # games yet, so pull the most recently finished season instead.
            events = completed_events(get_events(season=2025))
        if not events:
            return None

        results = []
        for e in events:
            comp = e["competitions"][0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            team_c = next((c for c in competitors
                            if c.get("team", {}).get("abbreviation", "").lower() == abbr), None)
            opp_c = next((c for c in competitors if c is not team_c), None)
            if not team_c or not opp_c:
                continue
            try:
                team_score = int(float(team_c.get("score", 0)))
                opp_score = int(float(opp_c.get("score", 0)))
            except (TypeError, ValueError):
                continue
            is_home = team_c.get("homeAway") == "home"
            win = team_score > opp_score
            opp_name = opp_c.get("team", {}).get("displayName", "Opponent")
            vs_at = "vs" if is_home else "@"
            result = (f"✅ W {team_score}-{opp_score} {vs_at} {opp_name}" if win
                      else f"❌ L {team_score}-{opp_score} {vs_at} {opp_name}")
            results.append({"date": e.get("date", ""), "result": result})

        results.sort(key=lambda x: x["date"], reverse=True)
        return results[:n] if results else None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def _fetch_espn_key_players_per_game(team_name, debug=False):
    """Top performers for a team's most recent completed regular-season game —
    passing/rushing/receiving only. This is the confirmed-working data source
    used directly by fetch_espn_key_players (a separate attempt at
    season-aggregate player stats was tried and confirmed not to exist in
    ESPN's public API — see fetch_espn_key_players's docstring).

    Defensive player leaders (tackles/sacks) were also attempted here and
    confirmed absent via live debugging across two different games — ESPN's
    per-game 'leaders' block only ever contains offensive categories, so
    defensive individual leaders genuinely aren't available from this data
    source at any level (game or season)."""
    try:
        abbr = ESPN_ABBR.get(team_name)
        if not abbr:
            return []

        def get_events(season=None):
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}/schedule"
            params = {"seasontype": 2}  # regular season only
            if season:
                params["season"] = season
            r = requests.get(url, params=params, timeout=10)
            return r.json().get("events", [])

        def most_recent_completed(events):
            completed = [e for e in events
                         if e.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed")]
            completed.sort(key=lambda e: e.get("date", ""), reverse=True)
            return completed[0] if completed else None

        # Confirmed via live debug (across two different games) that ESPN's
        # per-game 'leaders' block only ever contains passing/rushing/
        # receiving — no defensive categories exist in this data at all.
        wanted = [
            ("pass",    "🎯 Passing",       0),
            ("rush",    "🏃 Rushing",       1),
            ("receiv",  "🙌 Receiving",     2),
        ]

        def parse(event):
            if not event:
                return []
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            team_c = next((c for c in competitors
                            if c.get("team", {}).get("abbreviation", "").lower() == abbr), None)
            if debug:
                leaders_block = team_c.get("leaders", []) if team_c else []
                st.write(f"DEBUG (per-game fallback) — category names available for {team_name} (event {event.get('id')}):")
                st.json([cat.get("name", "?") for cat in leaders_block] if leaders_block
                         else {"note": "no matching competitor found, or no leaders block at all"})
            if not team_c:
                return []
            out = []
            found_labels = set()
            for cat in team_c.get("leaders", []):
                name = cat.get("name", "").lower()
                match = next(((label, order) for key, label, order in wanted
                              if key in name and label not in found_labels), None)
                if not match:
                    continue
                label, order = match
                lst = cat.get("leaders", [])
                if not lst:
                    continue
                top = lst[0]
                athlete = top.get("athlete", {})
                out.append({
                    "category": label,
                    "order": order,
                    "player": athlete.get("displayName", "Unknown"),
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "stat": top.get("displayValue", ""),
                    # Raw numeric value from ESPN (e.g. 198 for "198 YDS"),
                    # used for comparison since 'stat' is a compound string
                    # like "22/34, 198 YDS, 1 INT" that can't be parsed
                    # reliably for the number that actually matters.
                    "raw_value": top.get("value", None),
                })
                found_labels.add(label)
            return out

        events = get_events()
        event = most_recent_completed(events)
        out = parse(event)
        if not out:
            events = get_events(season=2025)
            event = most_recent_completed(events)
            out = parse(event)
        out.sort(key=lambda x: x["order"])
        return out
    except Exception:
        return []


def fetch_espn_key_players(team_name, debug=False):
    """Top performers from a team's most recent completed regular-season game.

    (An earlier attempt tried to find season-aggregate PLAYER stats —
    including defense — via teams/{abbr}/statistics. Live debugging
    confirmed that endpoint only returns TEAM-level season totals, not
    per-player data, so per-player season stats aren't available from ESPN's
    public API. That confirmed team-level data now powers the separate
    "Team Season Stats" section instead. This function is the per-game
    fallback that was already confirmed working, kept as the sole source.)"""
    return _fetch_espn_key_players_per_game(team_name, debug=debug)


@st.cache_data(ttl=3600)
def fetch_espn_team_standing(team_name, debug=False):
    """Team's current division/conference standing, pulled from ESPN's
    league-wide standings endpoint.

    (Earlier attempt used teams/{abbr} and its 'standingSummary' field —
    live debugging confirmed that field doesn't exist on this endpoint at
    all, full key list had no such field. Switched to the actual standings
    endpoint instead.)

    The exact response shape isn't confirmed live yet, so this recursively
    scans the JSON tree for any entry containing this team's abbreviation
    alongside a 'stats' list, and looks for a rank-like stat within it.
    Returns None on any failure or if nothing matching is found, so the UI
    can show a graceful fallback message. debug=True dumps what was found."""
    try:
        abbr = ESPN_ABBR.get(team_name)
        if not abbr:
            return None

        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/standings"
        r = requests.get(url, timeout=10)
        data = r.json()

        def find_team_entries(obj, found):
            if isinstance(obj, dict):
                team_info = obj.get("team")
                if isinstance(team_info, dict) and team_info.get("abbreviation", "").lower() == abbr:
                    found.append(obj)
                for v in obj.values():
                    find_team_entries(v, found)
            elif isinstance(obj, list):
                for item in obj:
                    find_team_entries(item, found)

        entries = []
        find_team_entries(data, entries)

        if debug:
            st.write(f"DEBUG (standing) — HTTP {r.status_code}, {len(entries)} matching entries found for {team_name}:")
            st.json(entries if entries else {"note": "no entry found containing this team's abbreviation + a 'team' key",
                                               "top_level_keys": list(data.keys()) if isinstance(data, dict) else None})

        for entry in entries:
            stats = entry.get("stats", [])
            for s in stats:
                name = s.get("name", "").lower()
                if "divisionrank" in name or name == "rank":
                    val = s.get("displayValue") or s.get("value")
                    if val:
                        return f"#{val} in division"
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_espn_team_injuries(team_name, debug=False):
    """Team's current injury report, pulled from ESPN's teams/{abbr}/injuries
    endpoint. NOT yet confirmed via live debugging — the exact response
    shape is unknown, so this tries a best-guess parse and returns [] if it
    doesn't match, so the UI can show a graceful fallback message instead of
    breaking. debug=True dumps the raw response to find the real shape."""
    try:
        abbr = ESPN_ABBR.get(team_name)
        if not abbr:
            return []
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}/injuries"
        r = requests.get(url, timeout=10)
        data = r.json()
        if debug:
            st.write(f"DEBUG (injuries) — HTTP {r.status_code}, raw response for {team_name}:")
            st.json(data)

        out = []
        # Best-guess shape: {"injuries": [{"injuries": [{"athlete": {...}, "status": ..., "details": {...}}]}]}
        for group in data.get("injuries", []):
            for item in group.get("injuries", []):
                athlete = item.get("athlete", {})
                out.append({
                    "player": athlete.get("displayName", "Unknown"),
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "status": item.get("status", ""),
                    "detail": item.get("shortComment", item.get("longComment", "")),
                })
        return out
    except Exception:
        return []


@st.cache_data(ttl=3600)
def fetch_espn_team_season_stats(team_name, debug=False):
    """Team-level season stat totals, pulled from ESPN's teams/{abbr}/statistics
    endpoint. Confirmed working via live debugging on 2025-09 — this endpoint
    returns aggregate TEAM totals (e.g. 'Chiefs passed for 3,927 yards this
    season'), not per-player stats, which is why fetch_espn_key_players uses
    a different data source for individual players.

    Matching is done PER-CATEGORY with EXACT category/stat names (the full
    category and stat name list was confirmed via live debugging), rather
    than flattening all stats into one global name->value dict, because some
    stat names are reused across categories with different meanings — e.g.
    'sacks' means sacks ALLOWED under 'passing' (offense being sacked) but
    sacks MADE under 'defensive' (opponents being sacked), and
    'interceptions' means INTs THROWN under 'passing' but INTs MADE under
    'defensiveInterceptions'. Flattening by name alone would let one
    silently overwrite the other.

    Returns [] on any failure so the UI can show a graceful fallback message
    instead of breaking."""
    try:
        abbr = ESPN_ABBR.get(team_name)
        if not abbr:
            return []

        def get_categories(season=None):
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}/statistics"
            params = {"season": season} if season else {}
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            categories = data.get("results", {}).get("stats", {}).get("categories", [])
            if debug:
                st.write(f"DEBUG (team season stats) — all category/stat names for {team_name} (season={season}):")
                overview = {cat.get("name", "?"): [s.get("name", "?") for s in cat.get("stats", [])]
                            for cat in categories}
                st.json(overview)
            return categories

        def find_stat(categories, category_name, stat_name):
            """Return the displayValue of stat_name within the category
            whose name exactly matches category_name (both confirmed real
            names from live debugging), or None if not found."""
            for cat in categories:
                if cat.get("name", "").lower() == category_name:
                    for s in cat.get("stats", []):
                        if s.get("name", "").lower() == stat_name:
                            return s.get("displayValue", "")
            return None

        # (exact category name, exact stat name, label, order) — all confirmed
        # real via live debugging of the full category/stat list.
        wanted = [
            ("passing",                 "totalpointspergame", "⭐ Points/Game",   0),
            ("passing",                 "passingyards",       "🎯 Pass Yards",    1),
            ("passing",                 "passingtouchdowns",  "🎯 Pass TDs",      2),
            ("rushing",                 "rushingyards",       "🏃 Rush Yards",    3),
            ("rushing",                 "rushingtouchdowns",  "🏃 Rush TDs",      4),
            ("passing",                 "interceptions",      "🧤 INTs Thrown",   5),
            ("passing",                 "sacks",              "💥 Sacks Allowed", 6),
            ("defensive",                "sacks",              "🛡️ Sacks Made",    7),
            ("defensiveinterceptions",  "interceptions",      "🧤 INTs Made",     8),
        ]

        def parse(categories):
            out = []
            for category_name, stat_name, label, order in wanted:
                val = find_stat(categories, category_name, stat_name)
                if val is not None:
                    out.append({"label": label, "value": val, "order": order})
            return out

        categories = get_categories()
        out = parse(categories)
        if not out:
            categories = get_categories(season=2025)
            out = parse(categories)
        out.sort(key=lambda x: x["order"])
        return out
    except Exception:
        return []




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
    # Only use games from the real held-back test set (X_test/y_test), grouped by season,
    # so accuracy reflects genuinely unseen games — not games the model was trained on.
    test_seasons = seasons.loc[X_test.index]
    lr_test_preds = lr.predict(X_test)
    season_acc=[]
    for s in sorted(test_seasons.unique()):
        if s<2000: continue
        mask = (test_seasons==s).values
        n_games = int(mask.sum())
        if n_games<5: continue
        season_acc.append({
            'season':int(s),
            'accuracy':accuracy_score(y_test[mask], lr_test_preds[mask]),
            'games': n_games,
            'note':season_notes.get(int(s),"")
        })

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

    scores_32=scores[scores['schedule_season']>=1990].copy()
    scores_32['period']=pd.cut(scores_32['schedule_season'],bins=[1989,1999,2009,2019,2026],labels=['1990–1999','2000–2009','2010–2019','2020–2025'])
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

PICKS_FILE = "nflnerd_picks.csv"

def load_picks():
    import os
    if os.path.exists(PICKS_FILE):
        return pd.read_csv(PICKS_FILE)
    return pd.DataFrame(columns=["timestamp","home_team","away_team","model_winner","model_prob","model_confidence","your_winner","your_confidence","agree"])

def save_pick(home_team, away_team, model_winner, model_prob, model_confidence, your_winner, your_confidence):
    df = load_picks()
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "home_team": home_team, "away_team": away_team,
        "model_winner": model_winner, "model_prob": f"{model_prob:.1%}",
        "model_confidence": model_confidence,
        "your_winner": your_winner, "your_confidence": your_confidence,
        "agree": "Yes" if your_winner == model_winner else "No"
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(PICKS_FILE, index=False)

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

# ── Separate blended model for LIVE predictions only ──
# This is intentionally kept apart from the Data Science tab's model.
# It blends 70% all-time history with 30% last-season-only stats, which
# was tested and verified to improve real accuracy (57.7% → 60.0%), keep
# all 6 weights football-sensible, and fix cases where the all-time-only
# model favoured a clearly weaker current team purely due to home
# advantage (e.g. Cardinals vs Rams). The Data Science tab intentionally
# continues to describe and use the original all-time-only model, since
# that content was independently finalized and verified against it.
@st.cache_resource
def load_predictor_model():
    scores_p = pd.read_csv('data/spreadspoke_scores.csv')
    scores_p = scores_p[(scores_p['score_home']>0)|(scores_p['score_away']>0)]
    renames = {
        'Oakland Raiders':'Las Vegas Raiders','St. Louis Rams':'Los Angeles Rams',
        'San Diego Chargers':'Los Angeles Chargers','Washington Redskins':'Washington Commanders',
        'Washington Football Team':'Washington Commanders','Tennessee Oilers':'Tennessee Titans',
        'Houston Oilers':'Tennessee Titans','Phoenix Cardinals':'Arizona Cardinals',
        'Baltimore Colts':'Indianapolis Colts','Los Angeles Raiders':'Las Vegas Raiders',
    }
    scores_p['team_home'] = scores_p['team_home'].replace(renames)
    scores_p['team_away'] = scores_p['team_away'].replace(renames)
    scores_p['home_win']  = (scores_p['score_home']>scores_p['score_away']).astype(int)
    scores_p['schedule_date'] = pd.to_datetime(scores_p['schedule_date'])
    scores_p = scores_p.sort_values('schedule_season')
    scores_p = scores_p[scores_p['schedule_season']>=1990].copy()

    def get_team_stats_blended(df, team, before_season, recent_weight=0.3):
        hg = df[(df['team_home']==team)&(df['schedule_season']<before_season)]
        ag = df[(df['team_away']==team)&(df['schedule_season']<before_season)]
        hw = (hg['score_home']>hg['score_away']).sum()
        aw = (ag['score_away']>ag['score_home']).sum()
        total = len(hg)+len(ag)
        if total==0: return 0.5,22.0,20.0
        all_time_wr = (hw+aw)/total
        all_time_scored   = pd.concat([hg['score_home'],ag['score_away']]).mean()
        all_time_conceded = pd.concat([hg['score_away'],ag['score_home']]).mean()

        recent_season = before_season - 1
        hg_r = hg[hg['schedule_season']==recent_season]
        ag_r = ag[ag['schedule_season']==recent_season]
        total_r = len(hg_r)+len(ag_r)
        if total_r==0:
            return all_time_wr, all_time_scored, all_time_conceded

        hw_r = (hg_r['score_home']>hg_r['score_away']).sum()
        aw_r = (ag_r['score_away']>ag_r['score_home']).sum()
        recent_wr = (hw_r+aw_r)/total_r
        recent_scored   = pd.concat([hg_r['score_home'],ag_r['score_away']]).mean()
        recent_conceded = pd.concat([hg_r['score_away'],ag_r['score_home']]).mean()

        blended_wr       = (1-recent_weight)*all_time_wr + recent_weight*recent_wr
        blended_scored   = (1-recent_weight)*all_time_scored + recent_weight*recent_scored
        blended_conceded = (1-recent_weight)*all_time_conceded + recent_weight*recent_conceded
        return blended_wr, blended_scored, blended_conceded

    game_data=[]
    for _,row in scores_p.iterrows():
        s=row['schedule_season']
        h_wr,h_sc,h_co=get_team_stats_blended(scores_p,row['team_home'],s)
        a_wr,a_sc,a_co=get_team_stats_blended(scores_p,row['team_away'],s)
        game_data.append({'home_win_rate':h_wr,'away_win_rate':a_wr,
            'home_avg_scored':h_sc,'away_avg_scored':a_sc,
            'home_avg_conceded':h_co,'away_avg_conceded':a_co,
            'home_win':row['home_win']})

    df_p = pd.DataFrame(game_data).dropna()
    X_p = df_p.drop('home_win', axis=1)
    y_p = df_p['home_win']
    lr_p = LogisticRegression(random_state=42, max_iter=1000)
    lr_p.fit(X_p, y_p)

    return lr_p, scores_p, get_team_stats_blended

# ── Load ─────────────────────────────────────
st.markdown('<p class="nflnerd-brand">🏈 NFLNerd</p>', unsafe_allow_html=True)
st.title("NFL Game Predictor")

with st.spinner("Loading model..."):
    (model,scores,get_team_stats,xgb_acc,lr_acc,rf_acc,
     xgb_f1,lr_f1,rf_f1,
     season_acc_df,cm,fpr,tpr,roc_auc,
     conf_data,feature_names,importances,
     X_test,y_test,period_hw,lr) = load_model()
    predictor_model, predictor_scores, predictor_get_team_stats = load_predictor_model()

accuracy = lr_acc
all_teams = sorted(CURRENT_NFL_TEAMS)

# ── Tabs ─────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
    "🔮 Predict",
    "📅 This Week's Games",
    "🏆 Team Rankings",
    "📜 Prediction History",
    "📊 How It Works",
    "🧪 Data Science",
    "🛠️ New Updates",
    "👤 Who Am I?"
])

# ════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════
with tab1:
    st.markdown("### 🔮 Predict Any NFL Matchup")
    st.markdown("**The NFL Game Predictor is a tool that predicts the outcome of any NFL matchup. Select the two teams and the model returns each team's probability of winning.**")
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
            hp,ap,conf,winner,h_wr,h_sc,h_co,a_wr,a_sc,a_co = predict_game(predictor_model,predictor_scores,predictor_get_team_stats,home_team,away_team)
            st.session_state['last_prediction'] = {
                'home_team':home_team,'away_team':away_team,'hp':hp,'ap':ap,'conf':conf,'winner':winner,
                'h_wr':h_wr,'h_sc':h_sc,'h_co':h_co,'a_wr':a_wr,'a_sc':a_sc,'a_co':a_co
            }
            st.session_state['pick_saved'] = False

    if 'last_prediction' in st.session_state:
            pred = st.session_state['last_prediction']
            home_team,away_team = pred['home_team'],pred['away_team']
            hp,ap,conf,winner = pred['hp'],pred['ap'],pred['conf'],pred['winner']
            h_wr,h_sc,h_co,a_wr,a_sc,a_co = pred['h_wr'],pred['h_sc'],pred['h_co'],pred['a_wr'],pred['a_sc'],pred['a_co']

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

            # Team standing — current division/conference position
            st.markdown("---")
            st.subheader("📊 Team Standing")
            c1,c2 = st.columns(2)
            home_standing = fetch_espn_team_standing(home_team)
            away_standing = fetch_espn_team_standing(away_team)
            with c1:
                st.markdown(f"**🏠 {home_team}**")
                st.markdown(home_standing if home_standing else "Standing unavailable right now.")
            with c2:
                st.markdown(f"**✈️ {away_team}**")
                st.markdown(away_standing if away_standing else "Standing unavailable right now.")
            if not home_standing and not away_standing:
                with st.expander("🛠️ Debug: inspect raw ESPN response (team standing)"):
                    st.caption("Shows every field ESPN's team endpoint returns, so the standing field can be found if it's named differently.")
                    fetch_espn_team_standing.clear()
                    fetch_espn_team_standing(home_team, debug=True)

            # Recent form — live from ESPN, falling back to historical CSV data if the API fails
            st.markdown("---")
            st.subheader("📅 Recent Form (Last 5 Games)")
            c1,c2=st.columns(2)
            with c1:
                st.markdown(f"**🏠 {home_team}**")
                home_form = fetch_espn_recent_form(home_team)
                if home_form:
                    for r in home_form: st.markdown(r['result'])
                else:
                    for _,r in get_recent_form(scores,home_team).iterrows(): st.markdown(r['result'])
            with c2:
                st.markdown(f"**✈️ {away_team}**")
                away_form = fetch_espn_recent_form(away_team)
                if away_form:
                    for r in away_form: st.markdown(r['result'])
                else:
                    for _,r in get_recent_form(scores,away_team).iterrows(): st.markdown(r['result'])

            # Team season stats — real season totals from ESPN (confirmed working)
            st.markdown("---")
            st.subheader("📈 Team Season Stats")
            st.caption("Season-long team totals so far. Higher value in green, lower in red, with the difference shown.")
            home_season_stats = fetch_espn_team_season_stats(home_team)
            away_season_stats = fetch_espn_team_season_stats(away_team)
            home_stat_dict = {s['label']: s['value'] for s in home_season_stats} if home_season_stats else {}
            away_stat_dict = {s['label']: s['value'] for s in away_season_stats} if away_season_stats else {}
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"**🏠 {home_team}**")
                if home_season_stats:
                    for s in home_season_stats:
                        home_html, _ = _compare_stat_pair(s['value'], away_stat_dict.get(s['label']))
                        st.markdown(f"{s['label']}: {home_html}", unsafe_allow_html=True)
                else:
                    st.caption("Season stats unavailable right now.")
            with c2:
                st.markdown(f"**✈️ {away_team}**")
                if away_season_stats:
                    for s in away_season_stats:
                        _, away_html = _compare_stat_pair(home_stat_dict.get(s['label']), s['value'])
                        st.markdown(f"{s['label']}: {away_html}", unsafe_allow_html=True)
                else:
                    st.caption("Season stats unavailable right now.")

            # Top performers from each team's most recent completed game
            st.markdown("---")
            st.subheader("🌟 Top Performers (Most Recent Game)")
            st.caption("Each team's top statistical performers from their last completed game. Higher value in green, lower in red, with the difference shown.")
            home_players = fetch_espn_key_players(home_team)
            away_players = fetch_espn_key_players(away_team)
            home_player_dict = {p['category']: p for p in home_players} if home_players else {}
            away_player_dict = {p['category']: p for p in away_players} if away_players else {}
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"**🏠 {home_team}**")
                if home_players:
                    for p in home_players:
                        away_p = away_player_dict.get(p['category'])
                        home_html, _ = _compare_stat_pair(
                            p['stat'], away_p['stat'] if away_p else None,
                            home_num=p['raw_value'], away_num=away_p['raw_value'] if away_p else None)
                        pos = f" ({p['position']})" if p['position'] else ""
                        st.markdown(f"{p['category']}: **{p['player']}**{pos} — {home_html}", unsafe_allow_html=True)
                else:
                    st.caption("Player stats unavailable right now.")
            with c2:
                st.markdown(f"**✈️ {away_team}**")
                if away_players:
                    for p in away_players:
                        home_p = home_player_dict.get(p['category'])
                        _, away_html = _compare_stat_pair(
                            home_p['stat'] if home_p else None, p['stat'],
                            home_num=home_p['raw_value'] if home_p else None, away_num=p['raw_value'])
                        pos = f" ({p['position']})" if p['position'] else ""
                        st.markdown(f"{p['category']}: **{p['player']}**{pos} — {away_html}", unsafe_allow_html=True)
                else:
                    st.caption("Player stats unavailable right now.")

            # Injury report — current injuries for both teams
            st.markdown("---")
            st.subheader("🏥 Injury Report")
            c1,c2 = st.columns(2)
            home_injuries = fetch_espn_team_injuries(home_team)
            away_injuries = fetch_espn_team_injuries(away_team)
            with c1:
                st.markdown(f"**🏠 {home_team}**")
                if home_injuries:
                    for inj in home_injuries[:8]:
                        pos = f" ({inj['position']})" if inj['position'] else ""
                        status = f" — {inj['status']}" if inj['status'] else ""
                        st.markdown(f"**{inj['player']}**{pos}{status}")
                        if inj['detail']:
                            st.caption(inj['detail'])
                else:
                    st.caption("No injuries reported right now — this is likely because injury reports typically aren't published until practice starts in the week of a game.")
            with c2:
                st.markdown(f"**✈️ {away_team}**")
                if away_injuries:
                    for inj in away_injuries[:8]:
                        pos = f" ({inj['position']})" if inj['position'] else ""
                        status = f" — {inj['status']}" if inj['status'] else ""
                        st.markdown(f"**{inj['player']}**{pos}{status}")
                        if inj['detail']:
                            st.caption(inj['detail'])
                else:
                    st.caption("No injuries reported right now — this is likely because injury reports typically aren't published until practice starts in the week of a game.")
            if not home_injuries and not away_injuries:
                with st.expander("🛠️ Debug: inspect raw ESPN response (injuries)"):
                    st.caption("Shows ESPN's raw injuries response, so the parsing can be corrected if the shape doesn't match.")
                    fetch_espn_team_injuries.clear()
                    fetch_espn_team_injuries(home_team, debug=True)

            st.markdown("---")
            st.subheader("🥊 Your NFLNerd Pick")
            st.markdown("Don't agree with the model? Set your own pick below, then click Save when you're ready.")
            c1, c2 = st.columns(2)
            with c1:
                your_winner = st.radio("Who do you think wins?", [home_team, away_team], key="your_winner_radio")
            with c2:
                your_confidence = st.radio("Your confidence", ["🔴 Low", "🟡 Medium", "🟢 High"], key="your_confidence_radio")

            if st.session_state.get('pick_saved'):
                st.success(f"✅ Pick saved — {your_winner} ({your_confidence}). Change your selections above and click Save again to log a new pick.")
            if st.button("💾 Save My Pick", use_container_width=True):
                save_pick(home_team, away_team, winner, hp if winner==home_team else ap, conf, your_winner, your_confidence)
                st.session_state['pick_saved'] = True
                st.rerun()

# ════════════════════════════════════════════
# TAB 2 — THIS WEEK'S GAMES
# ════════════════════════════════════════════
with tab2:
    st.markdown("### 📅 This Week's NFL Games")

    def format_kickoff(iso_str):
        """Format ESPN's UTC kickoff timestamp into a readable ET time.
        Uses a fixed UTC-4 offset (EDT), which is correct for the September
        portion of the season — would need adjusting for any games played
        after the US reverts to EST in early November."""
        if not iso_str:
            return ""
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ"):
            try:
                dt = datetime.strptime(iso_str, fmt)
                break
            except ValueError:
                continue
        else:
            return ""
        dt_et = dt - timedelta(hours=4)
        return dt_et.strftime("%a, %b %-d — %-I:%M %p ET")

    games, week_num, err = fetch_espn_scoreboard()

    if err or not games:
        st.info("🏈 **It's the offseason!** The NFL season hasn't started yet. This tab will automatically populate with live game predictions when the 2026 season kicks off in September. Check back then!")
        st.markdown("In the meantime, use the **🔮 Predict** tab to predict any hypothetical matchup.")
    else:
        week_label = f"NFL Week {week_num}" if week_num else "This Week's Games"
        hc1, hc2 = st.columns([4,1])
        with hc1:
            st.subheader(f"📅 {week_label}")
        with hc2:
            if st.button("🔄 Refresh Scores", use_container_width=True):
                fetch_espn_scoreboard.clear()
                st.rerun()

        # Map ESPN team names to our model's team names
        espn_to_model = {v.split('/')[-1].replace('.png',''):k for k,v in ESPN_LOGOS.items()}
        def match_team(espn_name):
            # Exact match first — ESPN's live scoreboard already returns the
            # exact same team name strings as CURRENT_NFL_TEAMS, confirmed
            # via live debugging. Checking this before the fuzzy substring
            # fallback below is essential: that fallback matches on
            # individual WORDS, and several team names share a common word
            # (e.g. "New England Patriots"/"New Orleans Saints"/"New York
            # Giants"/"New York Jets" all contain "New"; "Los Angeles Rams"/
            # "Los Angeles Chargers" both contain "Los"). Without checking
            # for an exact match first, whichever of those teams happened to
            # come first alphabetically in CURRENT_NFL_TEAMS would incorrectly
            # win the match every time — which is exactly what was happening
            # (Patriots and Chargers wrongly appearing in place of Saints/
            # Giants/Jets/Rams).
            if espn_name in CURRENT_NFL_TEAMS:
                return espn_name
            for model_name in CURRENT_NFL_TEAMS:
                if any(part.lower() in espn_name.lower() for part in model_name.split()):
                    return model_name
            return espn_name

        predictions = []
        for g in games:
            home = match_team(g['home'])
            away = match_team(g['away'])
            if home in CURRENT_NFL_TEAMS and away in CURRENT_NFL_TEAMS:
                hp,ap,conf,winner,_,_,_,_,_,_ = predict_game(predictor_model,predictor_scores,predictor_get_team_stats,home,away)
                predictions.append({**g,'home_mapped':home,'away_mapped':away,'hp':hp,'ap':ap,'conf':conf,'winner':winner})

        # Bye week note — any current team not appearing in this week's games.
        # Week 1 has no byes (all 32 teams play), so this naturally stays
        # silent until byes actually start later in the season.
        teams_playing = {p['home_mapped'] for p in predictions} | {p['away_mapped'] for p in predictions}
        bye_teams = sorted(set(CURRENT_NFL_TEAMS) - teams_playing)
        if bye_teams:
            st.info(f"💤 **On a bye this week:** {', '.join(bye_teams)}")

        if predictions:
            # Weekly highlights — based on the full week, not the filter below
            most_confident = max(predictions, key=lambda x: abs(x['hp']-x['ap']))
            closest = min(predictions, key=lambda x: abs(x['hp']-x['ap']))

            st.markdown("#### ⚡ NFLNerd Weekly Highlights")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**🟢 Most Confident Pick**")
                st.markdown(f"**{most_confident['winner']}** to win")
                st.markdown(f"{most_confident['home_mapped']} vs {most_confident['away_mapped']}")
                st.markdown(f"{max(most_confident['hp'],most_confident['ap']):.1%} probability")
            with c2:
                st.markdown("**🔴 Closest Game**")
                st.markdown(f"{closest['home_mapped']} vs {closest['away_mapped']}")
                st.markdown(f"{closest['hp']:.1%} vs {closest['ap']:.1%}")

            st.markdown("---")
            st.markdown("#### 🏈 Game Predictions")

            fc1, fc2 = st.columns(2)
            with fc1:
                team_filter = st.selectbox("Filter by team", ["All Teams"] + sorted(CURRENT_NFL_TEAMS), key="week_team_filter")
            with fc2:
                sort_option = st.selectbox("Sort by", ["Kickoff Time", "Model Confidence (High to Low)"], key="week_sort_option")

            display_games = predictions
            if team_filter != "All Teams":
                display_games = [g for g in display_games if team_filter in (g['home_mapped'], g['away_mapped'])]
            if sort_option == "Kickoff Time":
                display_games = sorted(display_games, key=lambda x: x.get('date',''))
            else:
                display_games = sorted(display_games, key=lambda x: abs(x['hp']-x['ap']), reverse=True)

            if not display_games:
                st.info(f"No game found for {team_filter} this week — likely a bye week.")

            with st.expander("🛠️ Debug: verify team record shows 0-0 before Week 1"):
                st.caption("Compares the record fetched with season=2026 (should be 0-0 before Week 1) against the same call with no season param (which previously returned each team's real 2025 record by mistake) — to confirm the fix actually worked.")
                if st.button("Check a team's record", key="record_debug_run"):
                    test_team = display_games[0]['home_mapped'] if display_games else CURRENT_NFL_TEAMS[0]
                    fetch_espn_team_record.clear()
                    fixed_record = fetch_espn_team_record(test_team, debug=True)
                    try:
                        abbr = ESPN_ABBR.get(test_team)
                        r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{abbr}", timeout=10)
                        old_team = r.json().get("team", {})
                        old_items = old_team.get("record", {}).get("items", [])
                        old_total = next((x for x in old_items if x.get("type")=="total"), None)
                        old_record = old_total.get("summary") if old_total else None
                    except Exception:
                        old_record = None
                    st.markdown(f"**{test_team}** — with `season=2026`: **{fixed_record}**  |  without season param: **{old_record}**")

            for g in display_games:
                diff = abs(g['hp']-g['ap'])
                conf_color = "#00C853" if diff>=0.15 else ("#FFB300" if diff>=0.07 else "#D50A0A")
                home_logo = ESPN_LOGOS.get(g['home_mapped'],'')
                away_logo = ESPN_LOGOS.get(g['away_mapped'],'')
                game_key = f"{g['home_mapped']}_{g['away_mapped']}_{g.get('date','')}".replace(" ","_")
                kickoff_str = format_kickoff(g.get('date',''))
                state = g.get('state', 'pre')

                with st.container():
                    st.markdown(f'<div class="match-card">', unsafe_allow_html=True)

                    if kickoff_str:
                        live_badge = ""
                        if state == "in":
                            live_badge = f" &nbsp;·&nbsp; <span style='color:#D50A0A;font-weight:700;'>🔴 LIVE — {g.get('status_detail','')}</span>"
                        elif state == "post":
                            live_badge = " &nbsp;·&nbsp; <span style='color:#888;font-weight:700;'>✅ FINAL</span>"
                        st.markdown(f"<div style='text-align:center;color:#888;font-size:13px;margin-bottom:8px;'>{kickoff_str}{live_badge}</div>", unsafe_allow_html=True)

                    c1,c2,c3,c4,c5 = st.columns([2,1,1,1,2])

                    # Once a game is live or final, show the real score instead of the
                    # predicted probability — the prediction is no longer the interesting number.
                    show_live_score = state in ("in", "post") and g.get('home_score') is not None and g.get('away_score') is not None

                    home_record = fetch_espn_team_record(g['home_mapped'])
                    away_record = fetch_espn_team_record(g['away_mapped'])

                    with c1:
                        if home_logo: st.image(home_logo, width=50)
                        st.markdown(f"**{g['home_mapped']}**" + (f" ({home_record})" if home_record else ""))
                        if show_live_score:
                            st.markdown(f"🏠 Home • **{g['home_score']}**")
                        else:
                            st.markdown(f"🏠 Home • {g['hp']:.1%}")
                    with c2:
                        st.markdown("<div style='text-align:center;padding-top:20px;color:#666;font-size:20px;'>VS</div>", unsafe_allow_html=True)
                    with c3:
                        if show_live_score:
                            actual_winner = g['home_mapped'] if float(g['home_score']) > float(g['away_score']) else (
                                            g['away_mapped'] if float(g['away_score']) > float(g['home_score']) else None)
                            if actual_winner:
                                winner_label = "🏠 HOME WIN" if actual_winner==g['home_mapped'] else "✈️ AWAY WIN"
                                st.markdown(f"<div style='text-align:center;padding-top:15px;'><span class='winner-badge'>{winner_label}</span></div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align:center;padding-top:15px;color:#888;'>TIED</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='text-align:center;color:#888;font-size:11px;margin-top:6px;'>Predicted: {g['winner']} ({max(g['hp'],g['ap']):.0%})</div>", unsafe_allow_html=True)
                        else:
                            winner_label = "🏠 HOME WIN" if g['winner']==g['home_mapped'] else "✈️ AWAY WIN"
                            st.markdown(f"<div style='text-align:center;padding-top:15px;'><span class='winner-badge'>{winner_label}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='text-align:center;color:{conf_color};font-size:12px;margin-top:6px;'>{g['conf']}</div>", unsafe_allow_html=True)
                    with c4:
                        st.markdown("<div style='text-align:center;padding-top:20px;color:#666;font-size:20px;'>VS</div>", unsafe_allow_html=True)
                    with c5:
                        if away_logo: st.image(away_logo, width=50)
                        st.markdown(f"**{g['away_mapped']}**" + (f" ({away_record})" if away_record else ""))
                        if show_live_score:
                            st.markdown(f"✈️ Away • **{g['away_score']}**")
                        else:
                            st.markdown(f"✈️ Away • {g['ap']:.1%}")

                    with st.expander("🥊 Make Your Own Pick"):
                        pc1, pc2 = st.columns(2)
                        with pc1:
                            your_winner = st.radio("Who do you think wins?", [g['home_mapped'], g['away_mapped']], key=f"your_winner_{game_key}")
                        with pc2:
                            your_confidence = st.radio("Your confidence", ["🔴 Low", "🟡 Medium", "🟢 High"], key=f"your_confidence_{game_key}")
                        if st.session_state.get(f"pick_saved_{game_key}"):
                            st.success(f"✅ Pick saved — {your_winner} ({your_confidence}). Change your selections above and save again to log a new pick.")
                        if st.button("💾 Save My Pick", key=f"save_{game_key}", use_container_width=True):
                            save_pick(g['home_mapped'], g['away_mapped'], g['winner'],
                                      g['hp'] if g['winner']==g['home_mapped'] else g['ap'],
                                      g['conf'], your_winner, your_confidence)
                            st.session_state[f"pick_saved_{game_key}"] = True
                            st.rerun()

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

    st.markdown("#### 🥊 Your Saved Picks vs the Model")
    st.markdown("Every time you've disagreed (or agreed) with the model on the Predict tab, it's logged here. These are hypothetical matchups you tested — not real upcoming games — so there's no actual result to check against, just a record of where your judgement and the model's line up or differ.")

    picks_df = load_picks()
    if picks_df.empty:
        st.info("No picks saved yet. Head to the 🔮 Predict tab, make a prediction, and save your own pick to see it here.")
    else:
        agree_count = (picks_df['agree']=='Yes').sum()
        disagree_count = (picks_df['agree']=='No').sum()
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Total Picks Logged", len(picks_df))
        with c2: st.metric("Agreed with Model", agree_count)
        with c3: st.metric("Disagreed with Model", disagree_count)
        st.dataframe(picks_df.sort_values('timestamp', ascending=False), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📅 Official Weekly Predictions")
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
    st.markdown("The NFL Game Predictor is a machine learning model trained and tested on over 9,000 games and 35 NFL seasons. This tab walks through everything behind the scenes - how it was built, what it learned, and how well it performs.")

    st.markdown("---")
    st.subheader("What is a Machine Learning Model (ML)?")
    st.markdown("""
An ML Model uses previous data to predict future outcomes. Similar to how a weather app studies years of past weather patterns to predict tomorrow's forecast - my model does the same, but with NFL games.
""")

    st.markdown("---")
    st.subheader("What is my ML Model trying to predict?")
    st.markdown("""
My model is trained on historical data to try and accurately predict which team will have the higher probability of winning any given NFL game. Rather than a simple yes or no, a probability is given - for example 64% win probability for the home team vs 36% win probability for the away team - so you can see how confident the model is in its prediction. How the model arrives at these probabilities will be explained later on.
""")

    st.markdown("---")
    st.subheader("What historical data is my model trained on?")
    st.markdown("""
The historical data comes from Kaggle - a website where data scientists share free datasets. The NFL dataset contains 9,455 game results, going back to 1990, giving the model plenty of data to be trained on, to make it as accurate as possible.

Before training, this data is split into two groups: about 7,500 games (80%) are used to actually train the model - this is where it learns the patterns of which teams are more likely to win the game (e.g. the team with the higher average points scored). The remaining 1,891 games (20%) are used as test games and are held back completely and never shown to the model during training. These are used afterwards in testing, to see how accurate the model really is, since it's being tested on games it has never seen before.
""")

    st.markdown("---")
    st.subheader("Why are there 6 key stats framed as home team vs away team?")
    st.markdown("""
The two teams in a matchup are distinguished as the home team and the away team. This is a major distinction that's always present between the two sides - true for every NFL game ever played.

From all of that historical NFL data, the model takes 6 key stats from each game - split into 3 for the home team and 3 for the away team, reflecting that home/away distinction. The same 6 stats are used for every single matchup, so every game is judged on the same criteria.
""")

    st.markdown("#### The 6 Stats used")
    st.markdown("""
1. **Home team historical win rate** - the percentage of all games the home team has won since 1990
2. **Away team historical win rate** - the percentage of all games the away team has won since 1990
3. **Home team average points scored** - how many points the home team scores per game on average since 1990
4. **Away team average points scored** - how many points the away team scores per game on average since 1990
5. **Home team average points conceded** - how many points the home team has conceded per game on average since 1990
6. **Away team average points conceded** - how many points the away team has conceded per game on average since 1990
""")

    total_games=len(scores); hw_pct=scores['home_win'].mean(); total_hw=int(scores['home_win'].sum())

    st.markdown("---")
    st.subheader("🏆 1. Which Model did I pick to power the NFL Game Predictor?")
    st.markdown("""
To find the best model to power the NFL Game Predictor, I tested 3 different ML models - Logistic Regression, Random Forest, and XGBoost - on how well they could correctly predict unseen NFL games, comparing their accuracy and F1 scores. My results showed that Logistic Regression came out on top for both accuracy and F1 score, so I chose it to power the NFL Game Predictor.
""")

    st.markdown("This bar chart shows the percentage of games each model correctly predicted, out of the 1,891 games held back for testing. The three models are closely matched, with less than 2% separating the best and worst performer, and only a 0.16% gap between Logistic Regression and XGBoost. All the models are between the 56-58% range, showing that each model is predicting games correctly more than half of the time, which is honestly quite good, considering how unpredictable the NFL is.")

    # Dynamic chart - automatically highlights whichever model actually scores highest
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

    st.markdown("#### F1 Scores")
    explainer("""
<b>What is an F1 score?</b><br>
Accuracy alone doesn't tell the whole story. If a model always predicted a home team win, no matter what - since home teams actually win around 57% of NFL games, this model would probably get 57 game predictions correct, purely by luck, giving it a 57% accuracy score, which is actually quite good. But this model would have a 0% accuracy at predicting away wins, and this wouldn't be captured by the accuracy score.<br><br>
This is exactly the problem the F1 score is designed to catch. Accuracy alone can make an unbalanced model (which only predicts the more likely outcome) look better than it really is. The F1 score checks how well a model predicts both home wins and away wins correctly, not just how often it's right overall.<br><br>

It combines two things:<br><br>

<b>Precision</b> - out of all the games the model predicted as a specific outcome (either a home win or an away win), how many of those predictions were actually correct?<br><br>

1. <b>Home Wins Example</b><br>
LR predicted 40 games as home wins. Out of those 40 predictions, 30 were correct (home wins) → Precision = 30 ÷ 40 = 75%<br><br>

2. <b>Away Wins Example</b><br>
LR predicted 25 games as away wins. Out of those 25 predictions, 15 were correct (away wins) → Precision = 15 ÷ 25 = 60%<br><br>

<b>Recall</b> - out of all the games that were actually a home win or an away win, how many did the model correctly predict?<br><br>

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
    st.subheader("How Does Logistic Regression Work?")
    explainer("""
<b>Step 1 - Assigning weights to the 6 stats</b><br>
Logistic Regression assigns a weight to each of the 6 key stats. It starts by guessing random weights - meaningless at first, since it has no idea of which stats are the most useful in predicting the winner. It then tests these weights against thousands of past games it already knows the real result of, to check how accurate its predictions are. Wherever its prediction is wrong, it nudges each weight slightly in the direction that would have made the prediction more accurate. This happens thousands of times, with tiny adjustments each time, until the weights settle into stable, final values – giving accurate predictions most of the time.<br><br>

<b>Step 2 - The intercept</b><br>
There is also one extra number called the intercept, which accounts for the home team's built-in advantage, shown by the home team winning 57% of games in the NFL. This means that even if the teams were even, the home team would still have the advantage due to this intercept.<br>
Home teams have an advantage due to:<br>
• <b>Crowd noise</b> - disrupts the away team's ability to communicate at the line of scrimmage<br>
• <b>No travel</b> - away teams often travel the day before, disrupting sleep and routine<br>
• <b>Stadium familiarity</b> - home teams know their own stadium, it's a new environment to the away team<br>
The intercept isn't manually set - it goes through the exact same training process as the 6 weights: starting as a random guess, then gradually adjusted through thousands of tiny corrections until it settles into a stable final value.<br><br>

<b>Step 3 - Combining everything into z</b><br>
Once training is complete, the intercept and all 6 weights are fixed - they never change again. Only the specific team stats change from matchup to matchup.<br>
The Z value is found using this formula:<br>
<b>z = intercept + (weight₁ × stat₁) + (weight₂ × stat₂) + (weight₃ × stat₃) + (weight₄ × stat₄) + (weight₅ × stat₅) + (weight₆ × stat₆)</b><br><br>
Z is just a stepping stone toward the final answer, and is not meaningful by itself<br><br>

<b>Step 4 - Turning z into an actual probability</b><br>
To turn z into a genuine, meaningful probability, it's run through a formula called the S-curve:<br>
<b>Probability of home win = 1 ÷ (1 + e^(-z))</b><br><br>
This formula is always built specifically to calculate the home team's probability of winning. The away team's probability is simply whatever's left over - 100% minus the home team's probability. This formula always produces a result between 0% and 100%, no matter what Z's value is.
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

    st.markdown("""
Some weights are positive and some are negative. A positive weight means that as that stat increases, the home team's chance of winning goes up. So if the Home Team Win Rate increases, the home team will have a better chance of winning - that's why it is a positive stat.
A negative weight means the opposite - as that stat goes up, the home team's chance of winning goes down. So if the Away Team Avg Points Scored increases, the home team will have a lower chance of winning, as it makes the away team stronger - that's why the stat is negative.
On the other hand, Away Team Avg Points Conceded is positive - because if the away team's defence concedes a lot, that benefits the home team's winning chances (they can score more).
""")

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
**Calculating z:**
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

**Turning z into a probability:**
```
Probability of Chiefs winning = 1 ÷ (1 + e^(-{ex_z:.4f})) = {ex_prob_home:.1%}
Probability of Eagles winning = {1-ex_prob_home:.1%}
```
So Logistic Regression predicts the Chiefs have a {ex_prob_home:.1%} chance of winning, and the Eagles have a {1-ex_prob_home:.1%} chance.
""")

    st.markdown("---")
    st.subheader("🔍 2. Which Stats Matter Most to the Model?")
    st.markdown("""
This chart uses the same real weights from the worked example above; these are the weights that Logistic Regression assigns to each of the 6 key stats. The bars show each weight and whether it's positive or negative (if it increases home team's win chance or if it decreases home team's win chance as the stat goes up). This helps see which stats have the biggest influence on the model's predictions.

**Reminder of what each key stat refers to**

1. Home team historical win rate - the percentage of all games the home team has won since 1990
2. Away team historical win rate - the percentage of all games the away team has won since 1990
3. Home team average points scored - how many points the home team scores per game on average since 1990
4. Away team average points scored - how many points the away team scores per game on average since 1990
5. Home team average points conceded - how many points the home team has conceded per game on average since 1990
6. Away team average points conceded - how many points the away team has conceded per game on average since 1990
""")
    idx=np.argsort(importances)[::-1]
    s_fn=[feature_names[i] for i in idx]
    s_fi=[importances[i] for i in idx]                 # magnitude, used for bar length/sorting
    s_signed=[lr.coef_[0][i] for i in idx]              # real signed weight, used for label + color
    s_colors=['#00C853' if v>=0 else '#D50A0A' for v in s_signed]  # green = positive, red = negative

    fig_fi=go.Figure(go.Bar(x=s_fi,y=s_fn,orientation='h',marker_color=s_colors,
        text=[f"{v:+.4f}" for v in s_signed],textposition='outside',textfont=dict(color='white')))
    fig_fi.update_layout(**CHART_LAYOUT,
        xaxis=dict(title='Weight Size (bar length = size, label = real signed weight)',gridcolor='#222',range=[0,max(s_fi)*1.35]),
        yaxis=dict(gridcolor='#222',autorange='reversed'),height=420,margin=dict(l=230,r=80,t=30,b=50))
    st.plotly_chart(fig_fi,use_container_width=True)
    st.caption("🟢 Green = positive weight (increases home team's win chance as the stat goes up)   🔴 Red = negative weight (decreases home team's win chance as the stat goes up)")

    st.markdown("""
From this graph, we can see that Away Team Win Rate has the single biggest weight of the 6, meaning it has the strongest pull on the model's predictions. If the away team's win rate is high, the home team's chance of winning drops significantly - which makes sense: a better away team is naturally harder to beat.

Then it's the Home Team Win Rate, and there is quite a big gap between Home vs Away Team Win Rate. This is because the home team already has a built-in advantage (playing at home), so they don't need a strong win rate to still be the favourites. The away team, however, has to overcome that disadvantage - so a good away win rate shows they're capable of winning both at home and on the road, whereas the home team doesn't need to prove that, since they already have home advantage against their opponent.

There's then a noticeable drop-off to the remaining 4 stats. Win rate is the most directly meaningful stat - a team that wins often clearly has a higher chance of winning again. Points scored and points conceded are less reliable indicators on their own, just because you have higher points scored/conceded doesn't necessarily mean that you're a better team than your opponent. A team with low average points conceded might still have a weak offence, and a team with high average points scored might just be racking up points in "garbage time" - scoring late against bench players once a game is already decided, rather than actually having a high powered offence. Whereas, with the win rate, it is much more accurate in showing how good a team is, as it considers the entire football game, both offense and defence and the winner.

The remaining 4 stats are all fairly close in weight, with a range of just 0.032. This makes sense, since points scored and points conceded are really just two sides of the same coin - one measures your own offence, the other measures your defence.
""")

    st.markdown("---")
    st.subheader("🎯 3. How Accurate Was the Model in Each NFL Season?")
    st.markdown("Hover over any point to see the exact accuracy and context for notable seasons.")

    if not season_acc_df.empty:
        avg_acc=season_acc_df['accuracy'].mean()

        # Bar chart FIRST - test games per season
        fig_games=go.Figure(go.Bar(
            x=season_acc_df['season'], y=season_acc_df['games'],
            marker_color='#555', text=season_acc_df['games'], textposition='outside', textfont=dict(color='white',size=10)))
        fig_games.update_layout(**CHART_LAYOUT,
            xaxis=dict(title='NFL Season',gridcolor='#222',dtick=2),
            yaxis=dict(title='Number of test games',gridcolor='#222'),
            height=280,margin=dict(t=20,b=50,l=60,r=40))
        st.plotly_chart(fig_games,use_container_width=True)

        st.markdown("""
The bar chart shows how many games from each season were used for testing. The amount ranges between roughly 40-70 games per season.
This matters for fairness, because a season with a significantly lower number of test games is less trustworthy - with fewer games to predict, luck plays a bigger role.
It could make the model look extremely accurate purely by chance. Or, it could cause the model to be quite inaccurate, as a small handful of unusual upset results could significantly drop the model's accuracy for that season.
A season with more test games doesn't have this problem - with more games to predict, individual upsets matter less, and the accuracy figure ends up being more precise, and not down to luck.
""")

        # Line graph SECOND - accuracy per season
        fig_s=go.Figure()
        fig_s.add_trace(go.Scatter(
            x=season_acc_df['season'],y=season_acc_df['accuracy']*100,
            mode='lines+markers',line=dict(color='#D50A0A',width=3),
            marker=dict(size=10,color='#D50A0A',line=dict(color='white',width=1.5)),
            customdata=np.stack([season_acc_df['note'], season_acc_df['games']], axis=-1),
            hovertemplate='<b>Season: %{x}</b><br>Accuracy: %{y:.1f}%<br>Test games: %{customdata[1]}<br>%{customdata[0]}<extra></extra>'))
        fig_s.add_hline(y=avg_acc*100,line_dash='dot',line_color='#4a90d9',
            annotation_text=f'Season average ({avg_acc:.1%})',annotation_font_color='#aaa',annotation_position='top right')
        fig_s.update_layout(**CHART_LAYOUT,
            xaxis=dict(title='NFL Season',gridcolor='#222',dtick=2,
                range=[season_acc_df['season'].min()-0.5,season_acc_df['season'].max()+0.5]),
            yaxis=dict(title='Games correctly predicted (%)',gridcolor='#222',dtick=5),
            height=460,margin=dict(t=50,b=60,l=70,r=80))
        st.plotly_chart(fig_s,use_container_width=True)

        st.markdown("""
This line graph shows the accuracy of each NFL season, it does this by presenting the percentage of that season's test games (the held-back 20%) that the model correctly predicted.
The graph does fluctuate from season to season. This is because every NFL season has upsets, injuries, and surprise teams - that's part of what makes the sport unpredictable in general.
But some seasons have a higher concentration of these disruptions than others, and some seasons have a lower concentration of these disruptions, which is what causes the accuracy to fluctuate.
The range isn't especially large, considering how unpredictable the NFL is known to be - every season sits between 47% and 64% accuracy.
This shows Logistic Regression is fairly consistent and accurate, reliably performing better than random guessing (50%) across almost every season.
""")

        st.markdown("#### The 2015 Outlier")
        st.markdown("""
2015 stands out as a clear outlier in the graph. This comes down to two things. First, the specific sample of games taken from the 2015 season happened to include an unusually high number of away wins - and since the model leans toward predicting home wins, this hurt its accuracy significantly. Second, it's common for any model like this to have some seasons that don't follow the usual pattern (home teams being favoured) - this is what causes anomalies.

It's also worth remembering this is a relatively small number of seasons - 25 in total. With a larger dataset spanning many more decades of NFL history, it's likely there would be more outliers like this - some seasons scoring even higher than 64%, and others dipping as low as 2015 did.
""")

        best=season_acc_df.loc[season_acc_df['accuracy'].idxmax()]
        worst=season_acc_df.loc[season_acc_df['accuracy'].idxmin()]
        c1,c2,c3=st.columns(3)
        with c1: st.metric("Season Average",f"{avg_acc:.1%}")
        with c2: st.metric("Best Season",f"{int(best['season'])} ({best['accuracy']:.1%}, {int(best['games'])} games)")
        with c3: st.metric("Toughest Season",f"{int(worst['season'])} ({worst['accuracy']:.1%}, {int(worst['games'])} games)")

    st.markdown("---")
    st.subheader("💪 4. Does the Model's Confidence Level Actually Mean Anything?")
    st.markdown("""
When the model predicts a game it gives each team a probability - for example home team 64%, away team 36%. However, the model will also give a confidence level. This is how confident the model is that its prediction will be accurate. The model decides its confidence based on how big the gap is between the two teams' probabilities.

**The three confidence levels:**
- 🔴 **Low Confidence** - This is when the gap between the winner and loser's probability is under 7% (e.g. 52% vs 48%). Could go either way.
- 🟡 **Medium Confidence** - This is when the gap between the winner and loser's probability is between 7 and 15% (e.g. 55% vs 45%). Leans one way.
- 🟢 **High Confidence** - This is when the gap between the winner and loser's probability is above 15% (e.g. 60% vs 40%). Strongly favours one team.
""")
    if conf_data:
        cdf=pd.DataFrame(conf_data)
        fig_c=go.Figure(go.Bar(x=cdf['confidence'],y=cdf['accuracy']*100,
            marker_color=['#D50A0A','#FFB300','#00C853'],
            text=[f"{a:.1%}\n({g} games)" for a,g in zip(cdf['accuracy'],cdf['games'])],
            textposition='outside',textfont=dict(color='white',size=13)))
        fig_c.add_hline(y=50,line_dash='dash',line_color='#555',annotation_text='Random guessing (50%)',annotation_font_color='#aaa',annotation_position='bottom right')
        fig_c.update_layout(**CHART_LAYOUT,yaxis=dict(title='Games correctly predicted (%)',range=[40,80],gridcolor='#222'),xaxis=dict(gridcolor='#222'),height=420)
        st.plotly_chart(fig_c,use_container_width=True)

        low_row=cdf[cdf['confidence'].str.contains('Low')]
        med_row=cdf[cdf['confidence'].str.contains('Medium')]
        high_row=cdf[cdf['confidence'].str.contains('High')]
        low_g = int(low_row['games'].values[0]) if not low_row.empty else 0
        med_g = int(med_row['games'].values[0]) if not med_row.empty else 0
        high_g = int(high_row['games'].values[0]) if not high_row.empty else 0

        st.markdown(f"""
From all 1,891 games the model predicted in testing, it also sorted them into different confidence buckets - low, medium, and high confidence. The number in brackets shows how many of those 1,891 games fell into each bucket: low confidence ({low_g} games), medium confidence ({med_g} games), and high confidence ({high_g} games). The percentage on each bar shows how accurate the model was within that confidence bucket.

Green is the highest, which makes sense - the model is highly confident in these predictions, so it should get more of them right, giving it the highest accuracy score. It also makes sense that the low confidence bucket has the lowest accuracy score, since those are the games the model is unsure about. This shows that the model is quite trustworthy - if it's confident in a prediction, it's right most of the time, and as it becomes more unsure (the red or yellow bucket), it starts to get more wrong.
""")

    st.markdown("---")
    st.subheader("🔢 5. Where Does the Model Go Wrong?")
    st.markdown("Below is a matrix of all the 1,891 test games, which the model has divided its predictions based on whether it predicted the home or the away team to win, and who actually won. Here are the results:")

    tn,fp,fn,tp=cm.ravel(); total_all=tn+tp+fp+fn; total_right=tn+tp
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

    st.markdown(f"""
Out of 1,891 test games, the model predicted the home team to win in {tp+fp:,} games and the away team in {tn+fn:,} games. Of those home predictions, {tp:,} were correct - an accuracy of {tp/(tp+fp):.1%}. Of the away predictions, {tn:,} were correct - an accuracy of {tn/(tn+fn):.1%}.

The model predicting home wins almost twice as often as away wins, combined with home predictions being more accurate, backs up the stat we found earlier - that home teams win around 57% of the time. This shows that the model leans heavily towards the home team, which is correct more often than not.

The amount of correctly predicted home games stands out significantly with {tp:,} games, really showing how much the home team wins, and how it is so disproportionate between the amount of home winner predictions and away winner predictions - almost doubled. That said, the {fp:,} incorrect home predictions show that home field advantage is not a guarantee. Even with the crowd, no travel, and stadium familiarity working in their favour, home teams still lost nearly 4 in 10 of the games the model backed them to win. On the other side, the {tn:,} correctly predicted away wins show that when an away team is genuinely superior - a higher win rate, better scoring, a stronger defence - the model will still back them to win, despite knowing they don't have home advantage. However, the {fn:,} incorrect away predictions could indicate that even a stronger away team can't always make up for the home team's advantage. Even so, this is the smallest quadrant, showing that if the away team is predicted to win, it is because they are a significantly stronger team than the home team, and that they do have a strong chance to win.

Overall, the model correctly predicted {total_right:,} of the 1,891 test games ({tp:,} home + {tn:,} away), which lines up with its overall accuracy of 57.7%. The {fp+fn:,} incorrect predictions ({fp:,} home, {fn:,} away) show the model is far from perfect - but a correct call more than half the time, on a sport as unpredictable as the NFL, still makes it a reliable tool.
""")

    st.markdown("---")
    st.subheader("📉 6. How Well Can the Model Separate Winners from Losers? (ROC Curve)")

    st.markdown("""
**What is a ROC curve?**

A ROC curve is a standard way in data science to measure how good the model is, overall, at telling home-win games apart from away-win games.

So far, every prediction we've looked at has used one fixed cutoff: if the model gives the home team a probability above 50%, it predicts a home win. The ROC curve tests something different - it imagines using many different cutoffs, from very strict (e.g. only call it a home win if the home team's win probability is over 90%) to very loose (e.g. call it a home win if the probability is barely over 51%), and checks how accurate the model is at each cutoff.

**The two axes**

For every cutoff tested, two things are measured:
- **True Positive Rate (Y-axis)** - out of all the games that were actually home wins, how many did the model correctly predict at this cutoff?
- **False Positive Rate (X-axis)** - out of all the games that were actually away wins, how many did the model incorrectly predict as home wins at this cutoff?

**Examples with different cutoffs**

Our test set has 1,077 real home wins and 814 real away wins. Here's how the X and Y coordinates are calculated at three different cutoffs:

**Cutoff = 90% (very strict)** - only games the model is extremely confident about get called a home win. A lot of home wins get missed (many only had 70-80% confidence), but the number of games predicted as a home win – and are incorrect – is very low, since the bar is so high.
- Correct home win predictions = 200 of 1,077 correct home wins → True Positive Rate = 200/1077 = 0.19
- Incorrect home win predictions (away teams actually won) = 5 of 814 incorrect home wins → False Positive Rate = 5/814 = 0.006
- Point plotted: (0.006, 0.19)

**Cutoff = 65% (moderate)** - more games clear the bar, so the model catches more home wins, but the number of games predicted as a home win – and are incorrect – starts increasing, since the cutoff is lower, the model isn't so certain the home team will win.
- Correct home win predictions = 756 of 1,077 home wins → True Positive Rate = 70.2%
- Incorrect home win predictions (away team actually won) = 481 of 814 real away wins → False Positive Rate = 59.1%
- Point plotted: (0.591, 0.702)

**Cutoff = 51% (very loose, barely above a coin flip)** - the model catches almost every home win, but also the number of games predicted as a home win – and are incorrect – is very high now, since it's now calling every close toss-up game as a home win.
- Correct home win predictions = 1,000 of 1,077 real home wins → True Positive Rate ≈ 0.93
- Incorrect home win predictions (away team actually won) = 500 of 814 real away wins → False Positive Rate ≈ 0.61
- Point plotted: (0.61, 0.93)

Each cutoff produces one point on the graph. Testing many different cutoffs - not just these three - and plotting a point for each one is what traces out the full ROC curve.

**Reading the curve**

Two reference lines help make sense of any ROC curve:
- A **perfect model** would catch every home win, at any cutoff, with zero incorrect home win predictions - its curve would shoot straight up to the top-left corner of the graph.
- A **useless model** - one with no real skill at predicting when the home team would win - would produce a straight diagonal line, since at every cutoff, it would be equally likely to catch a real home win as it would be to wrongly predict one, just a 50/50 chance either way.

A real, working model like ours sits somewhere between those two extremes - bulging above the diagonal line, but not reaching the perfect top-left corner.
""")

    fig_roc=go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',line=dict(color='#D50A0A',width=2),
        name=f'Our model (AUC = {roc_auc:.3f})',fill='tozeroy',fillcolor='rgba(213,10,10,0.08)'))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color='#666',dash='dash'),name='Random guessing (AUC = 0.5)'))
    fig_roc.update_layout(plot_bgcolor='#080808',paper_bgcolor='#080808',font=dict(color='#f0f0f0'),
        xaxis=dict(title='False Positive Rate →',gridcolor='#222',range=[0,1]),
        yaxis=dict(title='True Positive Rate →',gridcolor='#222',range=[0,1]),
        legend=dict(bgcolor='#111',bordercolor='#333'),height=440,showlegend=True)
    st.plotly_chart(fig_roc,use_container_width=True)

    st.markdown(f"""
**What is AUC?**

AUC stands for Area Under the Curve - how much of the graph's total area sits underneath the ROC curve.
- AUC = 0.5 → the curve sits exactly on the diagonal (useless, 50/50 decisions)
- AUC = 1.0 → the curve hugs the top-left corner (perfect model)
- Anything in between is what normal ML Models have, which can skilfully predict the winner, but is obviously not perfect

**What AUC means in practice**

AUC can be understood as a test: if you randomly compared one game that was a real home win, and a separate game that was a real away win, AUC is the probability that the model will make the home team's win probability higher in the game won by the home team, rather than the game won by the away team.

**Our model's real AUC score is {roc_auc:.3f}.**

This means: if you repeated that exact comparison - one real home win vs one separate real away win - many, many times, the model would correctly give the actual home-win game the higher probability {roc_auc:.1%} of the time (roughly {roc_auc*100:.0f} out of every 100 comparisons). If the AUC were exactly 0.5, that comparison would be a pure coin flip - correct only half the time.

An AUC of {roc_auc:.3f} confirms the model does have skill at predicting home wins - but it's by a modest amount.
""")

    st.markdown("---")
    st.subheader("🏠 7. Home field advantage")
    st.markdown("This section talks about how much advantage the home team has, and how it has influenced them winning across the years.")

    st.markdown("#### How was the overall home team win % calculated?")
    st.markdown(f"""
The overall home team win % will help us see how much the home field advantage actually helps the home team. If the home field advantage was non-existent, the overall home team win % should be 50%, equal with the away team. If the home field advantage is true, it should be above 50% and depending how much higher it is – should tell us how much advantage the home team has.

**{total_hw:,} home wins ÷ {total_games:,} total games = {hw_pct:.1%}**

57% shows us that the home field advantage is true, home teams do win more often than the away team, and that it is not equal.
""")

    st.markdown("**Reminder of the different types of home field advantage**")
    st.markdown("""
- Crowd noise - disrupts the away team's ability to communicate at the line of scrimmage
- No travel - away teams often travel the day before, disrupting sleep and routine
- Stadium familiarity - home teams know their own stadium; it's a new environment to the away team
""")

    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Total Games",f"{total_games:,}")
    with c2: st.metric("Seasons","1990–2025")
    with c3: st.metric("Total Home Wins",f"{total_hw:,}")
    with c4: st.metric("Home Win Rate",f"{hw_pct:.1%}")

    st.markdown("#### Has Home Field Advantage Changed Over Time?")
    st.markdown("""
This bar chart shows the different seasons since 1990, as that is how far the Kaggle data goes back to.

Seasons covered: 1990–1999 (10 seasons), 2000–2009 (10 seasons), 2010–2019 (10 seasons), 2020–2025 (6 seasons).

How each % was calculated: total home wins in that period ÷ total games in that period (shown on each bar).
""")
    bar_texts=[f"{r['Home Win Rate']:.1%}\n({int(r['Home Wins']):,} wins ÷ {int(r['Games']):,} games)" for _,r in period_hw.iterrows()]
    fig_p=go.Figure(go.Bar(x=period_hw['Period'],y=period_hw['Home Win Rate']*100,marker_color='#013369',
        text=bar_texts,textposition='outside',textfont=dict(color='white',size=12)))
    fig_p.add_hline(y=50,line_dash='dash',line_color='#555',annotation_text='50% - no home advantage',annotation_font_color='#aaa',annotation_position='bottom right')
    fig_p.update_layout(**CHART_LAYOUT,yaxis=dict(title='Home Win %',range=[0,75],gridcolor='#222'),xaxis=dict(gridcolor='#222'),height=400)
    st.plotly_chart(fig_p,use_container_width=True)

    st.markdown("""
From this bar chart, we can see that the home field advantage is slowly decreasing, this is because many away teams have actually adapted to the disadvantage they have, and have tried to mitigate it, as best as possible – this can be done by preparing for the home team's environment, making travel as fluid as possible or by adapting their communication so it's not affected by the loud crowd noise. 2020-2025 is especially low due to COVID, this is because the home team didn't have crowd noise – which is one of the main factors in the home field advantage. However, despite the home team win % decreasing, the advantage is still there, giving them an edge in the model's predictions, but in the future, the model may start to discover that the home field advantage is almost redundant, which may drastically change how the model starts predicting future games, even removing the intercept that gives the home team an edge in the model's predictions.
""")

    st.markdown("---")
    st.subheader("⚖️ 8. Evaluation & Future Improvements")
    st.markdown("""
Building this model has taught me a huge amount about machine learning - not just how to build one, but how to properly test it, question it, and understand where it genuinely struggles. This section is a look at the model's real limitations, and what I'd explore next if I kept developing it.

**1. Does data from 1990 still matter?**

The model is trained on data going back to 1990 - but the NFL has changed enormously since then. Rule changes, the rise of the passing game, free agency, salary cap rules, and even coaching styles have all shifted significantly over 35 years. A team's win rate from 1995 reflects a genuinely different era of football than a win in 2024, and it brings up the question - how much that older data actually helps the model? This is something worth testing further - comparing the model's performance using only recent seasons versus the full historical range.

**2. The model doesn't account for ties**

NFL ties are rare, but they do happen. Because the model is built entirely around one binary question - will the home team win? - there's no way for it to predict a tie, and games that ended in a tie were excluded from the dataset entirely during training. A more complete model would need to treat this as a three-outcome problem instead of two.

**3. No player-level data**

The model only knows team-level historical stats - win rate, points scored, points conceded. It has no way of knowing if a star quarterback is injured, if a key player was just traded, or if a team just hired a new head coach. A team's historical stats might look strong, but if their best player is unavailable, the model has no way to adjust for that. Incorporating injury reports or player-level data would be a meaningful next step.

**4. The model is noticeably weaker at predicting away wins**

Throughout this tab, we've seen consistent evidence that the model favours home teams more heavily than it should. It predicted a home win in 1,237 of 1,891 test games, but only an away win in 654 games - and its accuracy on home predictions (61.1%) is meaningfully higher than on away predictions (50.9%). This shows up again in the model's recall and precision: out of all the home wins, the model correctly predicted 70.2% of those home wins as well - but out of all the games the model predicted as home wins, only 61.1% actually were home wins. In other words, the model rarely misses a home win, since it's predicting a home win most of the time, but it's also calling a home win more often than it should, leading to the model being wrong almost 40% of the time.

**5. The 2015 season shows this limitation in action**

2015 was the model's worst-performing season by far - just 38% accuracy, compared to 47-64% in every other season. Investigating the actual game log showed this wasn't a bug: that specific batch of test games from 2015 had an unusually high number of away wins, and the model's built-in bias toward home teams meant it got a lower accuracy compared to its usual 47-64% range. This is a real example of the home team bias actually costing the model accuracy in a specific season.

**6. An experiment: using recent form instead of all-time history**

To test whether older data was "useless and ineffective" to the model, I rebuilt it using only each team's last 3 seasons of games, instead of their entire history since 1990. The results were mixed: accuracy actually improved slightly, from 57.7% to 59.6%. But the model's weights stopped making sense - Home Team Win Rate flipped to a negative weight, and Away Team Win Rate flipped positive, the opposite of what real football logic would suggest. As a stronger Home team Win rate helps the home team to win – so it should be a positive weight, and a weaker Away Team Win rate hurts the home team chances to win, so it should be a negative weight. This is likely because 3 seasons of data wasn't a large enough sample for the model to reliably give a weight to each individual stat.

Given this, I chose to keep the all-time historical model. A model that is slightly less accurate but can be explained why the model reaches its conclusions, is, in my opinion, much preferable to a model that's slightly more accurate but produces weights that don't make sense.
""")

# ════════════════════════════════════════════
# TAB 7 — NEW UPDATES
# ════════════════════════════════════════════
with tab7:
    st.markdown("### 🛠️ New Updates")
    st.markdown("A running changelog of improvements made to the NFL Game Predictor since the Data Science tab was finalized. Newest updates at the top.")
    st.markdown("---")

    st.markdown("#### 📅 [03/09/2026] — Blended model fixes home-advantage-only predictions")
    st.markdown("""
While building this NFL Predictor website, a real problem kept coming up when testing the Predict tab: for some matchups, the model would predict a team to win that anyone following the NFL right now would know is clearly the weaker team. For example, testing Arizona Cardinals (Home) vs Los Angeles Rams (Away) - the Rams are widely considered the best team in the NFL heading into this season, and the Cardinals one of the worst - but the model predicted the Cardinals to win, purely off home field advantage. This happens because all-time win rate, averaged across 35 years, doesn't reflect which teams are actually good right now. I did try to fix this in the past by replacing all-time historic data entirely with just the last 3 seasons, to make it more accurate to the current NFL, since how well a team did in the 1990s doesn't reflect how good they are now. The results were that it improved accuracy slightly, but the model's logic stopped making sense, so I reverted it.

So, I tried to improve the model: instead of replacing all-time data, I blended all-time data with data from the 2025 season - 70 (historic data)/30 (new data) split. This keeps the large, stable all-time data as the main source powering the predictor, while giving the model a meaningful nudge toward current team strength.

The results, tested the same way as before:

* Accuracy: 57.7% → 60.0%
* The Cardinals vs Rams case was fixed: Rams now correctly predicted to win (57.7% vs 42.3%), instead of the Cardinals

Since this genuinely passed every check, unlike the 3-season experiment, I kept it - but only for live predictions. The model described throughout Sections 1-6 of the Data Science tab is still the original, all-time-only model, since this is a newer update. However, the predict model above is what actually powers the Predict tab and This Week's Games tab.
""")

# ════════════════════════════════════════════
# TAB 8 — WHO AM I?
# ════════════════════════════════════════════
with tab8:
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
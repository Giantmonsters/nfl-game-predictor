"""
Live NFL data fetching.

Pulls the current season's schedule + results from nflverse and reshapes it
to match the columns in data/spreadspoke_scores.csv, so it can be appended
straight onto the historical dataset.

We fetch the raw CSV directly with pandas instead of using the nfl_data_py
package — that package pins an old version of pandas that fails to build on
modern Python (3.12+), and all it does under the hood is download this exact
file anyway. Fetching it ourselves avoids the dependency conflict entirely.
"""

import io
import requests
import pandas as pd

NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
FETCH_TIMEOUT_SECONDS = 10

# nflverse uses a few team abbreviations that differ from the ones in
# data/nfl_teams.csv (team_id column). Map nflverse -> our team_id.
ABBR_OVERRIDES = {
    "LA": "LAR",   # LA Rams
    "LV": "LVR",   # LV Raiders
}


def _team_id_to_name_map(teams_csv_path="data/nfl_teams.csv"):
    teams = pd.read_csv(teams_csv_path)
    # team_id can repeat across historical name changes (e.g. Raiders moved
    # cities) — keep the most recent team_name for each team_id by taking
    # the last row per id (file is roughly chronological).
    id_to_name = {}
    for _, row in teams.iterrows():
        id_to_name[row["team_id"]] = row["team_name"]
    return id_to_name


def _map_abbr(abbr, id_to_name):
    abbr = ABBR_OVERRIDES.get(abbr, abbr)
    return id_to_name.get(abbr, abbr)


def get_season_games(season, teams_csv_path="data/nfl_teams.csv"):
    """
    Returns (completed_games, upcoming_games) for the given season, both
    shaped like spreadspoke_scores.csv: schedule_date, schedule_season,
    schedule_week, team_home, score_home, score_away, team_away.

    completed_games: games that have a final score, ready to feed into
        team-stat calculations.
    upcoming_games: games with no score yet, for the "this week's
        matchups" prediction tab.

    Raises whatever exception the underlying fetch raises — callers should
    catch this and fall back to historical-only data so the app never goes
    down just because the data source is briefly unreachable.
    """
    id_to_name = _team_id_to_name_map(teams_csv_path)

    # Use requests with an explicit timeout instead of pandas.read_csv(url)
    # directly — pandas has no built-in timeout, so a stalled connection
    # (rather than a clean failure) can hang the app indefinitely.
    response = requests.get(NFLVERSE_GAMES_URL, timeout=FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    sched = pd.read_csv(io.StringIO(response.text))
    sched = sched[sched["season"] == season].copy()
    sched["team_home"] = sched["home_team"].apply(lambda a: _map_abbr(a, id_to_name))
    sched["team_away"] = sched["away_team"].apply(lambda a: _map_abbr(a, id_to_name))
    sched["schedule_date"] = pd.to_datetime(sched["gameday"])
    sched["schedule_season"] = sched["season"]
    sched["schedule_week"] = sched["week"]

    cols = ["schedule_date", "schedule_season", "schedule_week",
            "team_home", "score_home", "score_away", "team_away"]

    sched = sched.rename(columns={"home_score": "score_home", "away_score": "score_away"})

    has_score = sched["score_home"].notna() & sched["score_away"].notna()
    completed = sched.loc[has_score, cols].copy()
    upcoming = sched.loc[~has_score, cols].copy()

    return completed, upcoming


def get_current_week(upcoming_games):
    """Given the upcoming_games df, return the earliest week number still to be played."""
    if upcoming_games.empty:
        return None
    return int(upcoming_games["schedule_week"].min())

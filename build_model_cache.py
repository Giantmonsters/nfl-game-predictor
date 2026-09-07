"""
Run this ONCE, locally, to pre-train the models and save their results to
model_cache.pkl and predictor_cache.pkl. Commit BOTH .pkl files to your
GitHub repo alongside app.py.

Why: app.py normally retrains 3 ML models from scratch (LogReg, Random
Forest, XGBoost) plus a separate blended model, every time the app has a
cold start (e.g. after waking from sleep on Streamlit Cloud). That's slow.
With these cache files present in the repo, app.py loads the already-
trained results instantly instead of retraining — dramatically speeding up
cold starts.

IMPORTANT: The training logic below is a DELIBERATE, EXACT COPY of the
logic in app.py's load_model() and load_predictor_model() functions. If
you (or Claude) ever change how those functions train the model in
app.py, this script must be updated to match — otherwise the cached
results will silently drift out of sync with what app.py would compute on
its own. This was a necessary tradeoff to avoid a bigger restructuring of
app.py itself.

Usage:
    python build_model_cache.py

Requires: pandas, scikit-learn, xgboost (same as requirements.txt) and
your local data/spreadspoke_scores.csv file to be present, same as when
running the app itself.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
from xgboost import XGBClassifier

RENAMES = {
    'Oakland Raiders':'Las Vegas Raiders','St. Louis Rams':'Los Angeles Rams',
    'San Diego Chargers':'Los Angeles Chargers','Washington Redskins':'Washington Commanders',
    'Washington Football Team':'Washington Commanders','Tennessee Oilers':'Tennessee Titans',
    'Houston Oilers':'Tennessee Titans','Phoenix Cardinals':'Arizona Cardinals',
    'Baltimore Colts':'Indianapolis Colts','Los Angeles Raiders':'Las Vegas Raiders',
}


def get_team_stats(df, team, before_season):
    hg = df[(df['team_home']==team)&(df['schedule_season']<before_season)]
    ag = df[(df['team_away']==team)&(df['schedule_season']<before_season)]
    hw = (hg['score_home']>hg['score_away']).sum()
    aw = (ag['score_away']>ag['score_home']).sum()
    total = len(hg)+len(ag)
    if total==0: return 0.5,22.0,20.0
    return (hw+aw)/total, pd.concat([hg['score_home'],ag['score_away']]).mean(), pd.concat([hg['score_away'],ag['score_home']]).mean()


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


def build_main_model_cache():
    print("Building main model cache (model_cache.pkl)...")
    scores = pd.read_csv('data/spreadspoke_scores.csv')
    scores = scores[(scores['score_home']>0)|(scores['score_away']>0)]
    scores['team_home'] = scores['team_home'].replace(RENAMES)
    scores['team_away'] = scores['team_away'].replace(RENAMES)
    scores['home_win']  = (scores['score_home']>scores['score_away']).astype(int)
    scores['schedule_date'] = pd.to_datetime(scores['schedule_date'])
    scores = scores.sort_values('schedule_season')
    scores = scores[scores['schedule_season']>=1990].copy()

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
    importances=np.abs(lr.coef_[0])

    scores_32=scores[scores['schedule_season']>=1990].copy()
    scores_32['period']=pd.cut(scores_32['schedule_season'],bins=[1989,1999,2009,2019,2026],labels=['1990–1999','2000–2009','2010–2019','2020–2025'])
    period_hw=scores_32.groupby('period',observed=True)['home_win'].agg(home_win_rate='mean',games='count',home_wins='sum').reset_index()
    period_hw.columns=['Period','Home Win Rate','Games','Home Wins']

    season_acc_df = pd.DataFrame(season_acc)

    with open("model_cache.pkl", "wb") as f:
        pickle.dump({
            "lr": lr, "scores": scores,
            "xgb_acc": xgb_acc, "lr_acc": lr_acc, "rf_acc": rf_acc,
            "xgb_f1": xgb_f1, "lr_f1": lr_f1, "rf_f1": rf_f1,
            "season_acc_df": season_acc_df, "cm": cm, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
            "conf_data": conf_data, "feature_names": feature_names, "importances": importances,
            "X_test": X_test, "y_test": y_test, "period_hw": period_hw,
        }, f)
    print(f"  Saved model_cache.pkl — LR accuracy: {lr_acc:.1%}, F1: {lr_f1:.3f}")


def build_predictor_model_cache():
    print("Building predictor (blended) model cache (predictor_cache.pkl)...")
    scores_p = pd.read_csv('data/spreadspoke_scores.csv')
    scores_p = scores_p[(scores_p['score_home']>0)|(scores_p['score_away']>0)]
    scores_p['team_home'] = scores_p['team_home'].replace(RENAMES)
    scores_p['team_away'] = scores_p['team_away'].replace(RENAMES)
    scores_p['home_win']  = (scores_p['score_home']>scores_p['score_away']).astype(int)
    scores_p['schedule_date'] = pd.to_datetime(scores_p['schedule_date'])
    scores_p = scores_p.sort_values('schedule_season')
    scores_p = scores_p[scores_p['schedule_season']>=1990].copy()

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

    with open("predictor_cache.pkl", "wb") as f:
        pickle.dump({"lr_p": lr_p, "scores_p": scores_p}, f)
    print("  Saved predictor_cache.pkl")


if __name__ == "__main__":
    build_main_model_cache()
    build_predictor_model_cache()
    print("\nDone! Now commit both model_cache.pkl and predictor_cache.pkl to your repo.")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load data
scores = pd.read_csv('data/spreadspoke_scores.csv')

# Only use games where we have scores (completed games)
scores = scores[(scores['score_home'] > 0) | (scores['score_away'] > 0)]

# Create target variable - 1 if home team wins, 0 if away team wins
scores['home_win'] = (scores['score_home'] > scores['score_away']).astype(int)

# Sort by date
scores = scores.sort_values('schedule_season')

# Calculate rolling team stats
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

print("Building features... this may take a minute!")

# Only use data from 1990 onwards for speed
scores = scores[scores['schedule_season'] >= 1990].copy()

game_data = []
for _, row in scores.iterrows():
    season = row['schedule_season']
    home_team = row['team_home']
    away_team = row['team_away']
    
    home_wr, home_scored, home_conceded = get_team_stats(scores, home_team, season)
    away_wr, away_scored, away_conceded = get_team_stats(scores, away_team, season)
    
    game_data.append({
        'home_win_rate': home_wr,
        'away_win_rate': away_wr,
        'home_avg_scored': home_scored,
        'away_avg_scored': away_scored,
        'home_avg_conceded': home_conceded,
        'away_avg_conceded': away_conceded,
        'home_win': row['home_win']
    })

df_features = pd.DataFrame(game_data)
df_features = df_features.dropna()

print(f"Features built! Total games: {len(df_features)}")

# Split features and target
X = df_features.drop('home_win', axis=1)
y = df_features['home_win']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
print("\nTraining models...")

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression accuracy: {lr_acc:.1%}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest accuracy: {rf_acc:.1%}")

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
print(f"XGBoost accuracy: {xgb_acc:.1%}")

print("\nDone!")

# Prediction function
def predict_game(home_team, away_team):
    home_wr, home_scored, home_conceded = get_team_stats(scores, home_team, 2026)
    away_wr, away_scored, away_conceded = get_team_stats(scores, away_team, 2026)
    
    game_features = [[home_wr, away_wr, home_scored, away_scored, home_conceded, away_conceded]]
    
    prob = xgb.predict_proba(game_features)[0]
    home_prob = prob[1]
    away_prob = prob[0]
    
    print(f"\n🏈 {home_team} (Home) vs {away_team} (Away)")
    print(f"{home_team} win probability: {home_prob:.1%}")
    print(f"{away_team} win probability: {away_prob:.1%}")
    
    if home_prob > away_prob:
        print(f"Predicted winner: {home_team} 🏆")
    else:
        print(f"Predicted winner: {away_team} 🏆")

# Test it!
predict_game("Kansas City Chiefs", "Philadelphia Eagles")
predict_game("Buffalo Bills", "Miami Dolphins")
predict_game("Dallas Cowboys", "New York Giants")
predict_game("San Francisco 49ers", "Los Angeles Rams")
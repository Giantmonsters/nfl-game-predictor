import pandas as pd

# Load the data
scores = pd.read_csv('data/spreadspoke_scores.csv')
teams = pd.read_csv('data/nfl_teams.csv')

# Basic info
print("=== SCORES DATASET ===")
print(f"Shape: {scores.shape}")
print(f"\nColumns: {list(scores.columns)}")
print(f"\nFirst 5 rows:")
print(scores.head())

print("\n=== TEAMS DATASET ===")
print(f"Shape: {teams.shape}")
print(f"\nColumns: {list(teams.columns)}")
print(f"\nFirst 5 rows:")
print(teams.head())

# Check date range
print(f"\nSeason range: {scores['schedule_season'].min()} to {scores['schedule_season'].max()}")

# Check for missing values
print(f"\nMissing values:")
print(scores.isnull().sum())

# Check score distribution
print(f"\nAverage home score: {scores['score_home'].mean():.1f}")
print(f"Average away score: {scores['score_away'].mean():.1f}")
print(f"Home win rate: {(scores['score_home'] > scores['score_away']).mean():.1%}")
import pandas as pd
import numpy as np
from scipy.stats import mstats
from sklearn.preprocessing import RobustScaler
import joblib

def load_team_stats():
    """Load and process team stats from CSV with correct column names"""
    team_stats = pd.read_csv('nba_team_stats_2024_2025.csv')
    team_stats['Team'] = team_stats['Team'].str.replace('*', '').str.strip()
    return team_stats[['Team', 'DEF_FG%', 'DEF_3P%', 'DEF_PTS']]

def calculate_advanced_metrics(player_df):
    """Enhance player data with advanced metrics"""
    # Load team defensive stats
    team_stats = load_team_stats()
    
    # Extract opponent from the "OPPONENT" column
    player_df['OPPONENT'] = player_df['OPPONENT'].str.split().str[-1]
    
    # Merge with team stats
    player_df = pd.merge(
        player_df,
        team_stats,
        left_on='OPPONENT',
        right_on='Team',
        how='left'
    )
    
    # Convert MIN column to minutes (handle both int and MM:SS formats)
    def convert_minutes(min_value):
        if isinstance(min_value, str) and ':' in min_value:
            # Handle MM:SS format
            minutes, seconds = min_value.split(':')
            return int(minutes) + int(seconds) / 60
        elif isinstance(min_value, (int, float)):
            # Handle integer or float (assume it's already in minutes)
            return min_value
        else:
            # Handle unexpected formats (e.g., NaN or invalid strings)
            return 0  # or np.nan if you prefer to handle missing values differently
    
    player_df['MIN'] = player_df['MIN'].apply(convert_minutes)
    
    # Calculate advanced metrics
    player_df['TS%'] = player_df['PTS'] / (2 * (player_df['FGA'] + 0.44 * player_df['FTA']))
    
    return player_df.drop(columns=['Team'])

def preprocess_data():
    # Load and process data
    players = pd.read_csv('nba_player_stats_2024_2025.csv')
    players = calculate_advanced_metrics(players)
    
    # Winsorize outliers
    for metric in ['PTS', 'REB', 'AST']:
        players[f'{metric}_WINSOR'] = mstats.winsorize(players[metric], limits=[0.01, 0.01])
    
    # Define features using available columns
    features = [
        'PTS_WINSOR', 'REB_WINSOR', 'AST_WINSOR',
        'DEF_FG%', 'DEF_3P%', 'TS%', 'MIN'
    ]
    
    # Scale features
    scaler = RobustScaler()
    players[features] = scaler.fit_transform(players[features])
    joblib.dump(scaler, 'advanced_scaler.pkl')
    
    # Create targets
    for stat in ['PTS', 'REB', 'AST']:
        players[f'NEXT_{stat}'] = players.groupby('PLAYER_NAME')[stat].shift(-1)

    # Replace dropna() with fillna()
    players.fillna(0).to_csv('preprocessed_data_advanced.csv', index=False)
    print("✅ Preprocessing complete")

if __name__ == '__main__':
    preprocess_data()
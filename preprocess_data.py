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
    
    # Convert MIN column to minutes
    def convert_minutes(min_value):
        if isinstance(min_value, str) and ':' in min_value:
            minutes, seconds = min_value.split(':')
            return int(minutes) + int(seconds)/60
        return float(min_value)
    
    player_df['MIN'] = player_df['MIN'].apply(convert_minutes)
    
    # Calculate advanced metrics
    player_df['TS%'] = player_df['PTS'] / (2 * (player_df['FGA'] + 0.44 * player_df['FTA'] + 1e-6))
    
    # Winsorize stats
    for metric in ['PTS', 'REB', 'AST']:
        winsorized = mstats.winsorize(player_df[metric], limits=[0.01, 0.01])
        player_df[f'{metric}_WINSOR'] = winsorized
    
    # Add trend features
    player_df['LAST_3_PTS_AVG'] = player_df.groupby('PLAYER_NAME')['PTS'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    player_df['LAST_5_PTS_TREND'] = player_df.groupby('PLAYER_NAME')['PTS'].transform(
        lambda x: x.rolling(5).mean().pct_change(fill_method=None)
    )
    
    # Handle infinity/NaN
    player_df['LAST_5_PTS_TREND'] = player_df['LAST_5_PTS_TREND'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return player_df.drop(columns=['Team'])
def preprocess_data():
    # Load and process data
    players = pd.read_csv('nba_player_stats_2024_2025.csv')
    players = calculate_advanced_metrics(players)

    # Define features using available columns
    features = [ 'PTS_WINSOR', 'LAST_3_PTS_AVG', 'LAST_5_PTS_TREND',
    'REB_WINSOR', 'AST_WINSOR', 
    'DEF_FG%', 'DEF_3P%', 'TS%', 'MIN']
    
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
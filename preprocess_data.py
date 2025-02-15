import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler

def preprocess_data():
    # Load and clean data
    data = pd.read_csv('nba_player_stats_2024_2025.csv')
    data.columns = data.columns.str.strip()
    
    # Convert and validate dates
    data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'], format='%b %d, %Y', errors='coerce')
    data = data.dropna(subset=['GAME_DATE'])
    
    # Sort by PLAYER_NAME and GAME_DATE (most recent first)
    data = data.sort_values(['PLAYER_NAME', 'GAME_DATE'], ascending=[True, False])
    
    # Calculate advanced metrics (retain original values)
    data['EFF'] = (data['PTS'] + data['REB'] + data['AST'] + data['STL'] + data['BLK'] 
                   - (data['FGA'] - data['FGM']) - (data['FTA'] - data['FTM']) - data['TOV'])
    
    # Winsorize outliers (1% on both ends)
    metrics = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    for metric in metrics:
        data[f'{metric}_WINSOR'] = data.groupby('PLAYER_NAME')[metric].transform(
            lambda x: stats.mstats.winsorize(x, limits=[0.01, 0.01]))
    
    # Time-weighted features (exponential moving average)
    for player in data['PLAYER_NAME'].unique():
        player_mask = data['PLAYER_NAME'] == player
        for metric in ['PTS', 'REB', 'AST']:
            # EMA and rolling median
            data.loc[player_mask, f'EMA_{metric}'] = (
                data.loc[player_mask, f'{metric}_WINSOR'].ewm(alpha=0.3, adjust=False).mean())
            data.loc[player_mask, f'ROLLING_MED_{metric}'] = (
                data.loc[player_mask, f'{metric}_WINSOR'].rolling(5, min_periods=1).median())
    
    # Add 5-game rolling average (original PTS)
    data['LAST_5_AVG_PTS'] = data.groupby('PLAYER_NAME')['PTS'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Define features to scale (use winsorized versions to reduce outlier impact)
    features_to_scale = [
        'PTS_WINSOR', 'REB_WINSOR', 'AST_WINSOR', 'STL_WINSOR', 'BLK_WINSOR', 'TOV_WINSOR',
        'EMA_PTS', 'EMA_REB', 'EMA_AST', 'ROLLING_MED_PTS', 'ROLLING_MED_REB', 'ROLLING_MED_AST',
        'LAST_5_AVG_PTS', 'MIN'
    ]
    
    # Scale features for model training
    scaler = RobustScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    
    # Create targets (NEXT_PTS, NEXT_REB, NEXT_AST) using original values
    for metric in ['PTS', 'REB', 'AST']:
        data[f'NEXT_{metric}'] = data.groupby('PLAYER_NAME')[metric].shift(-1)
    
    # Remove rows with missing targets
    data = data.dropna(subset=['NEXT_PTS', 'NEXT_REB', 'NEXT_AST'])
    
    # Save preprocessed data (retain original stats for display)
    data.to_csv('preprocessed_data.csv', index=False)
    print("✅ Preprocessed data saved")

if __name__ == '__main__':
    preprocess_data()
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
    
    # Sort by PLAYER_NAME and GAME_DATE in descending order (most recent games first)
    data = data.sort_values(['PLAYER_NAME', 'GAME_DATE'], ascending=[True, False])
    
    # Calculate advanced metrics
    data['EFF'] = (data['PTS'] + data['REB'] + data['AST'] + data['STL'] + data['BLK'] 
                   - (data['FGA'] - data['FGM']) - (data['FTA'] - data['FTM']) - data['TOV'])
    
    # Winsorize outliers (top/bottom 1%)
    metrics = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    for metric in metrics:
        data[metric] = data.groupby('PLAYER_NAME')[metric].transform(
            lambda x: stats.mstats.winsorize(x, limits=[0.01, 0.01])
        )
    
    # Time-weighted features
    for player in data['PLAYER_NAME'].unique():
        player_mask = data['PLAYER_NAME'] == player
        for metric in ['PTS', 'REB', 'AST']:
            # Exponential moving average
            data.loc[player_mask, f'EMA_{metric}'] = (
                data.loc[player_mask, metric].ewm(alpha=0.3, adjust=False).mean()
            )
            # Rolling median
            data.loc[player_mask, f'ROLLING_MED_{metric}'] = (
                data.loc[player_mask, metric].rolling(5, min_periods=1).median()
            )

    # Add 5-game rolling averages
    data['LAST_5_AVG_PTS'] = data.groupby('PLAYER_NAME')['PTS'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Create targets with outlier filtering
    for metric in ['PTS', 'REB', 'AST']:
        data[f'NEXT_{metric}'] = data.groupby('PLAYER_NAME')[metric].shift(-1)
        q1 = data[f'NEXT_{metric}'].quantile(0.05)
        q3 = data[f'NEXT_{metric}'].quantile(0.95)
        data = data[(data[f'NEXT_{metric}'] >= q1) & (data[f'NEXT_{metric}'] <= q3)]
    
    # Normalize/Scale features
    features = [
        'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'FG_PCT', 'FG3_PCT', 'FT_PCT', 'EFF',
        'EMA_PTS', 'EMA_REB', 'EMA_AST',
        'ROLLING_MED_PTS', 'ROLLING_MED_REB', 'ROLLING_MED_AST',
        'LAST_5_AVG_PTS'
    ]
    scaler = RobustScaler()
    data[features] = scaler.fit_transform(data[features])
    
    # Save preprocessed data
    data.to_csv('preprocessed_data.csv', index=False)
    print("✅ Preprocessed data saved")

if __name__ == '__main__':
    preprocess_data()
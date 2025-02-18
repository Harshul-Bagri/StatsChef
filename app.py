import streamlit as st
import pandas as pd
import numpy as np
import joblib
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from scipy.stats import mstats
import time

# Configure page
st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")

# Load models and scaler
@st.cache_resource
def load_resources():
    return {
        'models': {
            'NEXT_PTS': joblib.load('final_model_NEXT_PTS.pkl'),
            'NEXT_REB': joblib.load('final_model_NEXT_REB.pkl'),
            'NEXT_AST': joblib.load('final_model_NEXT_AST.pkl')
        },
        'scaler': joblib.load('robust_scaler.pkl')
    }

resources = load_resources()
models = resources['models']
scaler = resources['scaler']

# Function to check if a player is active
def is_player_active(player_id):
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        info_df = player_info.get_data_frames()[0]
        return info_df['ROSTERSTATUS'].iloc[0] == "Active"
    except Exception as e:
        print(f"Error checking player status for ID {player_id}: {e}")
        return False

# Sidebar controls
st.sidebar.title("Controls")
player_name = st.sidebar.text_input('Enter Player Name', '')

def fetch_player_data(player_name):
    """Fetch game log for selected player using NBA API"""
    # Find player ID
    all_players = players.get_players()
    player_dict = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
    if not player_dict:
        return None
    player_id = player_dict[0]['id']
    
    # Check if player is active
    if not is_player_active(player_id):
        return None
    
    # Fetch current season data
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id, 
        season='2024-25'  # Update season as needed
    )
    return gamelog.get_data_frames()[0]

def preprocess_data(raw_df):
    """Process raw API data into model-ready features"""
    if raw_df.empty:
        return None, None
    
    # Convert minutes to float and sort chronologically
    raw_df = raw_df.copy()
    
    def convert_minutes(x):
        if isinstance(x, str) and ':' in x:
            minutes, seconds = x.split(':')
            return int(minutes) + int(seconds)/60
        elif isinstance(x, (int, float)):
            return float(x)
        else:
            return 0.0

    raw_df['MIN'] = raw_df['MIN'].apply(convert_minutes)
    raw_df['GAME_DATE'] = pd.to_datetime(raw_df['GAME_DATE'])
    raw_df = raw_df.sort_values('GAME_DATE', ascending=True)  # Oldest first
    
    # Calculate Efficiency (EFF)
    raw_df['EFF'] = (
        raw_df['PTS'] + raw_df['REB'] + raw_df['AST'] + 
        raw_df['STL'] + raw_df['BLK'] - 
        (raw_df['FGA'] - raw_df['FGM']) - 
        (raw_df['FTA'] - raw_df['FTM']) - raw_df['TOV']
    )
    
    # Winsorize stats (1% on each end)
    metrics = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    for metric in metrics:
        raw_df[f'{metric}_WINSOR'] = mstats.winsorize(raw_df[metric], limits=[0.01, 0.01])
    
    # Time-based features
    for metric in ['PTS', 'REB', 'AST']:
        # Exponential Moving Average
        raw_df[f'EMA_{metric}'] = raw_df[f'{metric}_WINSOR'].ewm(alpha=0.3, adjust=False).mean()
        # Rolling median (5 games)
        raw_df[f'ROLLING_MED_{metric}'] = raw_df[f'{metric}_WINSOR'].rolling(5, min_periods=1).median()
    
    # 5-game average points
    raw_df['LAST_5_AVG_PTS'] = raw_df['PTS'].rolling(5, min_periods=1).mean()
    
    # Sort by recent games first
    raw_df = raw_df.sort_values('GAME_DATE', ascending=False)
    
    # Prepare features for scaling
    features = [
        'PTS_WINSOR', 'REB_WINSOR', 'AST_WINSOR', 'STL_WINSOR', 'BLK_WINSOR', 'TOV_WINSOR',
        'EMA_PTS', 'EMA_REB', 'EMA_AST', 'ROLLING_MED_PTS', 'ROLLING_MED_REB', 'ROLLING_MED_AST',
        'LAST_5_AVG_PTS', 'MIN'
    ]
    
    # Scale features
    try:
        scaled_features = scaler.transform(raw_df[features])
    except Exception as e:
        st.error(f"Scaling error: {e}")
        return None, None
    
    return raw_df, scaled_features[0]

# Main app logic
if player_name:
    raw_data = fetch_player_data(player_name)
    if raw_data is None:
        st.error("Player not found or not active!")
        st.stop()

    # Preprocess the data
    latest_raw, scaled_features = preprocess_data(raw_data)
    if scaled_features is None:
        st.error("Error processing data!")
        st.stop()

    # Display latest game stats
    st.title(f"🏀 {player_name}'s Performance")
    st.header("📊 Latest Game Stats")

    # Use the preprocessed data (latest_raw) instead of raw_data
    latest_game = latest_raw.iloc[0]
    st.write(f"**Date:** {pd.to_datetime(latest_game['GAME_DATE']).strftime('%Y-%m-%d')}")

    stats = {
        'Points': latest_game['PTS'],
        'Rebounds': latest_game['REB'],
        'Assists': latest_game['AST'],
        'Steals': latest_game['STL'],
        'Blocks': latest_game['BLK'],
        'Efficiency': latest_game['EFF'],  # Now available
        '5-Game Avg': latest_raw['LAST_5_AVG_PTS'].iloc[0],
        'Season Avg': latest_raw['PTS'].mean()
    }
    st.table(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))

    # Predictions
    st.header("🔮 Next Game Projections")
    input_data = np.array(scaled_features).reshape(1, -1)

    col1, col2, col3 = st.columns(3)
    with col1:
        pts_pred = models['NEXT_PTS'].predict(input_data)[0]
        st.metric("Predicted Points", f"{pts_pred:.1f}")
    with col2:
        reb_pred = models['NEXT_REB'].predict(input_data)[0]
        st.metric("Predicted Rebounds", f"{reb_pred:.1f}")
    with col3:
        ast_pred = models['NEXT_AST'].predict(input_data)[0]
        st.metric("Predicted Assists", f"{ast_pred:.1f}")
else:
    st.info("Please enter a player's name in the sidebar to get started.")
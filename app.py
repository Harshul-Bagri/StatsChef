import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
from io import StringIO
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from scipy.stats import mstats

# Configure page
st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")

# Define features globally
FEATURES = [
    'PTS_WINSOR', 'LAST_3_PTS_AVG', 'LAST_5_PTS_TREND',
    'REB_WINSOR', 'AST_WINSOR', 
    'DEF_FG%', 'DEF_3P%', 'TS%', 'MIN'
]

# Load models and scaler
@st.cache_resource
def load_resources():
    return {
        'models': {
            'NEXT_PTS': joblib.load('advanced_model_NEXT_PTS.pkl'),
            'NEXT_REB': joblib.load('advanced_model_NEXT_REB.pkl'),
            'NEXT_AST': joblib.load('advanced_model_NEXT_AST.pkl')
        },
        'scaler': joblib.load('advanced_scaler.pkl')
    }

resources = load_resources()
models = resources['models']
scaler = resources['scaler']

def is_player_active(player_id):
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        info_df = player_info.get_data_frames()[0]
        return info_df['ROSTERSTATUS'].iloc[0] == "Active"
    except Exception as e:
        print(f"Error checking player status for ID {player_id}: {e}")
        return False

def scrape_defensive_stats(team_abbreviation):
    team_map = {
        'HOU': 'Houston Rockets', 'SAC': 'Sacramento Kings',
        'UTA': 'Utah Jazz', 'GSW': 'Golden State Warriors',
        'LAL': 'Los Angeles Lakers', 'BOS': 'Boston Celtics',
        'BKN': 'Brooklyn Nets', 'PHI': 'Philadelphia 76ers',
        'MIL': 'Milwaukee Bucks', 'CHI': 'Chicago Bulls',
        'DET': 'Detroit Pistons', 'IND': 'Indiana Pacers',
        'CLE': 'Cleveland Cavaliers', 'ATL': 'Atlanta Hawks',
        'MIA': 'Miami Heat', 'CHA': 'Charlotte Hornets',
        'ORL': 'Orlando Magic', 'WAS': 'Washington Wizards',
        'DEN': 'Denver Nuggets', 'MIN': 'Minnesota Timberwolves',
        'OKC': 'Oklahoma City Thunder', 'POR': 'Portland Trail Blazers',
        'DAL': 'Dallas Mavericks', 'SAS': 'San Antonio Spurs',
        'MEM': 'Memphis Grizzlies', 'NOP': 'New Orleans Pelicans',
        'PHX': 'Phoenix Suns', 'LAC': 'Los Angeles Clippers',
        'NYK': 'New York Knicks', 'TOR': 'Toronto Raptors'
    }
    
    try:
        full_name = team_map[team_abbreviation]
        response = requests.get("https://www.basketball-reference.com/leagues/NBA_2025.html")
        soup = BeautifulSoup(response.content, 'html.parser')
        def_table = soup.find('table', {'id': 'per_game-opponent'})
        
        if not def_table:
            st.error("Defensive stats table not found.")
            return None
            
        def_df = pd.read_html(StringIO(str(def_table)))[0]
        def_df['Team'] = def_df['Team'].str.replace('*', '', regex=False)
        team_stats = def_df[def_df['Team'] == full_name]
        
        return {
            'DEF_FG%': team_stats.iloc[0]['FG%'],
            'DEF_3P%': team_stats.iloc[0]['3P%'],
            'DEF_PTS': team_stats.iloc[0]['PTS']
        } if not team_stats.empty else None
        
    except Exception as e:
        st.error(f"Error scraping stats: {str(e)}")
        return None

# Sidebar controls
st.sidebar.title("Controls")
player_name = st.sidebar.text_input('Enter Player Name', '')

def fetch_player_data(player_name):
    all_players = players.get_players()
    player_dict = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
    if not player_dict:
        return None
    player_id = player_dict[0]['id']
    
    if not is_player_active(player_id):
        return None
    
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
    return gamelog.get_data_frames()[0]

def preprocess_data(raw_df, opponent_stats):
    if raw_df.empty:
        return None, None
    
    raw_df = raw_df.copy()
    
    # Convert minutes
    def convert_minutes(x):
        try:
            if isinstance(x, str) and ':' in x:
                mins, secs = x.split(':')
                return float(mins) + float(secs)/60
            return float(x)
        except:
            return 0.0

    raw_df['MIN'] = raw_df['MIN'].apply(convert_minutes)
    
    # Date handling
    raw_df['GAME_DATE'] = pd.to_datetime(raw_df['GAME_DATE'], format='%b %d, %Y', errors='coerce')
    raw_df = raw_df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
    
    # Advanced metrics
    denominator = 2 * (raw_df['FGA'] + 0.44 * raw_df['FTA'] + 1e-6)
    raw_df['TS%'] = np.where(denominator > 0, raw_df['PTS'] / denominator, 0.0)
    
    # Winsorization
    for metric in ['PTS', 'REB', 'AST']:
        raw_df[metric] = raw_df[metric].fillna(0)
        raw_df[f'{metric}_WINSOR'] = mstats.winsorize(raw_df[metric], limits=[0.01, 0.01])
    
    # Trend features
    raw_df['LAST_3_PTS_AVG'] = raw_df['PTS'].rolling(3, min_periods=1).mean()
    raw_df['LAST_5_PTS_TREND'] = raw_df['PTS'].rolling(5).mean().pct_change(fill_method=None)
    raw_df['LAST_5_PTS_TREND'] = raw_df['LAST_5_PTS_TREND'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Opponent stats
    for stat in ['DEF_FG%', 'DEF_3P%', 'DEF_PTS']:
        raw_df[stat] = opponent_stats.get(stat, 0)
    
    # Final cleanup
    raw_df[FEATURES] = raw_df[FEATURES].fillna(0)
    
    if raw_df[FEATURES].isnull().any().any():
        st.error("NaN values detected after preprocessing")
        st.stop()
    
    try:
        scaled_features = scaler.transform(raw_df[FEATURES])
        scaled_features = np.nan_to_num(scaled_features)
        return raw_df, scaled_features[0]
    except Exception as e:
        st.error(f"Scaling failed: {str(e)}")
        return None, None

# Main app logic
if player_name:
    raw_data = fetch_player_data(player_name)
    if raw_data is None or raw_data.empty:
        st.error("Player not found or no data available!")
        st.stop()

    latest_game = raw_data.iloc[0]
    opponent = latest_game['MATCHUP'].split()[-1]
    opponent_stats = scrape_defensive_stats(opponent)
    
    if not opponent_stats:
        st.error(f"Could not fetch stats for {opponent}")
        st.stop()
    
    latest_raw, scaled_features = preprocess_data(raw_data, opponent_stats)
    
    if scaled_features is None:
        st.error("Data processing failed")
        st.stop()
    
    # Display stats
    st.title(f"🏀 {player_name}'s Performance")
    st.header("📊 Latest Game Stats")
    game_date = latest_raw.iloc[0]['GAME_DATE']
    st.write(f"**Date:** {game_date.strftime('%Y-%m-%d') if not pd.isna(game_date) else 'Unknown'}")
    
    stats = {
        'Points': latest_raw.iloc[0]['PTS'],
        'Rebounds': latest_raw.iloc[0]['REB'],
        'Assists': latest_raw.iloc[0]['AST'],
        'True Shooting %': latest_raw.iloc[0]['TS%'],
        'Opponent': opponent
    }
    st.table(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))
    
    # Predictions
    st.header("🔮 Next Game Projections")
    input_data = pd.DataFrame([scaled_features], columns=FEATURES)
    
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
    
    st.subheader("Why This Projection?")
    st.write(f"Opponent **{opponent}** allows {opponent_stats['DEF_PTS']} PPG")
    
else:
    st.info("Please enter a player's name in the sidebar to get started.")
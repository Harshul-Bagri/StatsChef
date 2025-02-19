import streamlit as st
import pandas as pd
import joblib
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# Load resources
@st.cache_resource
def load_resources():
    return {
        'models': {target: joblib.load(f'advanced_model_{target}.pkl') 
                  for target in ['NEXT_PTS', 'NEXT_REB', 'NEXT_AST']},
        'scaler': joblib.load('advanced_scaler.pkl'),
        'team_stats': pd.read_csv('nba_team_stats_2024_2025.csv')
    }

def get_opponent_stats(opponent_name):
    """Get defensive stats for opponent team"""
    team_stats = st.session_cache['team_stats']
    opponent_stats = team_stats[team_stats['Team'] == opponent_name].iloc[0]
    return {
        'DEF_RATING': opponent_stats['DEF_RATING'],
        'DEF_FG%': opponent_stats['DEF_FG%'],
        'DEF_PTS': opponent_stats['DEF_PTS']
    }

def main():
    st.set_page_config(page_title="NBA Predictor Pro", layout="wide")
    resources = load_resources()
    
    # Player selection
    player_name = st.sidebar.selectbox(
        'Select Player',
        sorted([p['full_name'] for p in players.get_players()])
    )
    
    if player_name:
        # Get player data
        player = [p for p in players.get_players() if p['full_name'] == player_name][0]
        gamelog = playergamelog.PlayerGameLog(player['id'], season='2024-25').get_data_frames()[0]
        
        # Get opponent stats
        latest_game = gamelog.iloc[0]
        opponent = latest_game['MATCHUP'].split()[-1]
        def_stats = get_opponent_stats(opponent)
        
        # Prepare features
        features = pd.DataFrame([{
            'PTS_WINSOR': latest_game['PTS'],
            'REB_WINSOR': latest_game['REB'],
            'AST_WINSOR': latest_game['AST'],
            'DEF_RATING': def_stats['DEF_RATING'],
            'DEF_FG%': def_stats['DEF_FG%'],
            'TS%': latest_game['PTS'] / (2 * (latest_game['FGA'] + 0.44 * latest_game['FTA'])),
            'MIN': int(latest_game['MIN'].split(':')[0]) + int(latest_game['MIN'].split(':')[1])/60
        }])
        
        # Make predictions
        predictions = {
            target: resources['models'][target].predict(features)[0]
            for target in ['NEXT_PTS', 'NEXT_REB', 'NEXT_AST']
        }
        
        # Display results
        st.title(f"{player_name} Next Game Projection")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Points", f"{predictions['NEXT_PTS']:.1f}")
        with col2:
            st.metric("Predicted Rebounds", f"{predictions['NEXT_REB']:.1f}")
        with col3:
            st.metric("Predicted Assists", f"{predictions['NEXT_AST']:.1f}")

if __name__ == '__main__':
    main()
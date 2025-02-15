import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure page
st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    return pd.read_csv('preprocessed_data.csv')  # Contains original stats

@st.cache_resource
def load_models():
    return {
        'NEXT_PTS': joblib.load('final_model_NEXT_PTS.pkl'),
        'NEXT_REB': joblib.load('final_model_NEXT_REB.pkl'),
        'NEXT_AST': joblib.load('final_model_NEXT_AST.pkl')
    }

data = load_data()
models = load_models()

# Sidebar
st.sidebar.title("Controls")
player_name = st.sidebar.selectbox(
    'Select Player',
    options=sorted(data['PLAYER_NAME'].unique()),
    index=0
)

# Display latest game stats (original values)
def get_latest_stats(player_name):
    player_data = data[data['PLAYER_NAME'] == player_name]
    return player_data.iloc[0] if not player_data.empty else None

latest_stats = get_latest_stats(player_name)

if latest_stats is not None:
    st.title(f"🏀 {player_name}'s Performance")
    
    # Latest game stats (original values)
    st.header("📊 Latest Game Stats")
    st.write(f"**Date:** {pd.to_datetime(latest_stats['GAME_DATE']).strftime('%d/%m/%Y')}")
    
    stats = {
        'Points': latest_stats['PTS'],
        'Rebounds': latest_stats['REB'],
        'Assists': latest_stats['AST'],
        'Steals': latest_stats['STL'],
        'Blocks': latest_stats['BLK'],
        'Efficiency': latest_stats['EFF'],
        '5-Game Avg': latest_stats['LAST_5_AVG_PTS'],
        'Season Avg': data[data['PLAYER_NAME'] == player_name]['PTS'].mean()
    }
    
    st.table(pd.DataFrame.from_dict(stats, orient='index', columns=['Value']))
    
    # Next game projections (using scaled features)
    st.header("🔮 Next Game Projections")
    
    # Prepare input features (scaled/winsorized)
    features = [
        'PTS_WINSOR', 'REB_WINSOR', 'AST_WINSOR', 'STL_WINSOR', 'BLK_WINSOR', 'TOV_WINSOR',
        'EMA_PTS', 'EMA_REB', 'EMA_AST', 'ROLLING_MED_PTS', 'ROLLING_MED_REB', 'ROLLING_MED_AST',
        'LAST_5_AVG_PTS', 'MIN'
    ]
    
    input_data = np.array([latest_stats[f] for f in features]).reshape(1, -1)
    
    # Get predictions
    pts_pred = models['NEXT_PTS'].predict(input_data)[0]
    reb_pred = models['NEXT_REB'].predict(input_data)[0]
    ast_pred = models['NEXT_AST'].predict(input_data)[0]
    
    # Display predictions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Points", f"{pts_pred:.1f}")
    with col2:
        st.metric("Rebounds", f"{reb_pred:.1f}")
    with col3:
        st.metric("Assists", f"{ast_pred:.1f}")

else:
    st.error("Player not found. Please try another name.")
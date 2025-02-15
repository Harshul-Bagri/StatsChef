import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure page
st.set_page_config(
    page_title="NBA Performance Predictor",
    page_icon="🏀",
    layout="wide"
)
#The App is hostel on streamlit for now
# Load resources
@st.cache_data
def load_data():
    return pd.read_csv('preprocessed_data.csv')

@st.cache_resource
def load_models():
    return joblib.load('final_models.pkl')

data = load_data()
models = load_models()

# Sidebar
st.sidebar.title("Controls")
player_name = st.sidebar.selectbox(
    'Select Player',
    options=sorted(data['PLAYER_NAME'].unique()),
    index=sorted(data['PLAYER_NAME'].unique()).index('LeBron James') if 'LeBron James' in data['PLAYER_NAME'].unique() else 0
)

# Main content
st.title("🏀 NBA Player Performance Predictor")
st.markdown("### Next Game Projections")

def get_latest_stats(player_name):
    player_data = data[data['PLAYER_NAME'] == player_name]
    return player_data.iloc[0] if not player_data.empty else None

latest_stats = get_latest_stats(player_name)

if latest_stats is not None:
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("### 📊 Latest Game Stats")
        latest_game_date = pd.to_datetime(latest_stats['GAME_DATE']).strftime('%d/%m')
        st.markdown(f"**Date:** {latest_game_date}")
        
        # Create stats table with safe feature access
        stats_data = {
            'Points': latest_stats.get('PTS', np.nan),
            'Rebounds': latest_stats.get('REB', np.nan),
            'Assists': latest_stats.get('AST', np.nan),
            'Steals': latest_stats.get('STL', np.nan),
            'Blocks': latest_stats.get('BLK', np.nan),
            'Efficiency': latest_stats.get('EFF', np.nan),
            '5-Game Avg': latest_stats.get('LAST_5_AVG_PTS', np.nan),
            'Season Avg': data[data['PLAYER_NAME'] == player_name]['PTS'].mean()
        }
        
        stats_table = pd.DataFrame({
            'Stat': list(stats_data.keys()),
            'Value': list(stats_data.values())
        }).set_index('Stat')
        
        st.dataframe(
            stats_table.style.format({'Value': '{:.1f}'}),
            use_container_width=True
        )

    with col2:
        st.markdown("### 🔮 Next Game Projections")
        
        # Safely prepare input features
        required_features = [
            'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'EFF',
            'EMA_PTS', 'EMA_REB', 'EMA_AST',
            'ROLLING_MED_PTS', 'ROLLING_MED_REB', 'ROLLING_MED_AST',
            'LAST_5_AVG_PTS'
        ]
        
        input_features = np.array([
            latest_stats.get(f, 0) for f in required_features
        ]).reshape(1, -1)

        # Safe prediction with error handling
        try:
            pts_pred, pts_intervals = models['NEXT_PTS'].predict(input_features, alpha=0.05)
            reb_pred, reb_intervals = models['NEXT_REB'].predict(input_features, alpha=0.05)
            ast_pred, ast_intervals = models['NEXT_AST'].predict(input_features, alpha=0.05)
            
            # Convert numpy types to Python floats
            predictions = {
                'Points': (float(pts_pred[0]), 
                            [float(pts_intervals[0][0]), float(pts_intervals[0][1])]),
                'Rebounds': (float(reb_pred[0]), 
                            [float(reb_intervals[0][0]), float(reb_intervals[0][1])]),
                'Assists': (float(ast_pred[0]), 
                            [float(ast_intervals[0][0]), float(ast_intervals[0][1])])
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            # Fallback predictions without confidence intervals
            pts_pred = models['NEXT_PTS'].predict(input_features)
            reb_pred = models['NEXT_REB'].predict(input_features)
            ast_pred = models['NEXT_AST'].predict(input_features)
            
            predictions = {
                'Points': (float(pts_pred[0]), [float(pts_pred[0]-2), float(pts_pred[0]+2)]),
                'Rebounds': (float(reb_pred[0]), [float(reb_pred[0]-1), float(reb_pred[0]+1)]),
                'Assists': (float(ast_pred[0]), [float(ast_pred[0]-1), float(ast_pred[0]+1)])
            }

        # Display predictions
        pred_cols = st.columns(3)
        for i, (stat, (value, interval)) in enumerate(predictions.items()):
            with pred_cols[i]:
                st.markdown(f"""
                <div style="
                    background: #f0f2f6;
                    padding: 1rem;
                    border-radius: 10px;
                    text-align: center;
                    margin: 0.5rem;
                ">
                    <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">{stat}</h4>
                    <h2 style="margin: 0; color: #e74c3c;">{value:.1f}</h2>
                    <div style="color: #7f8c8d; font-size: 0.9rem;">
                        {interval[0]:.1f} - {interval[1]:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Trends visualization
    st.markdown("---")
    st.markdown("### 📈 Performance Trends (Last 30 Games)")
    
    player_data = data[data['PLAYER_NAME'] == player_name].tail(30)
    if not player_data.empty:
        player_data = player_data.copy()
        player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE']).dt.strftime('%d/%m')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        for stat, color in zip(['PTS', 'REB', 'AST'], ['#3498db', '#2ecc71', '#9b59b6']):
            if stat in player_data.columns:
                sns.lineplot(
                    data=player_data,
                    x='GAME_DATE',
                    y=stat,
                    label=stat,
                    color=color,
                    linewidth=2.5,
                    marker='o'
                )
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Game Date (DD/MM)", labelpad=15)
        plt.ylabel("Stat Value", labelpad=15)
        plt.ylim(bottom=0)
        plt.title(f"{player_name}'s Performance Trend", pad=20)
        plt.legend(title='Statistics', loc='upper left')
        st.pyplot(fig)
    else:
        st.warning("Insufficient data for trend visualization")

else:
    st.error("Player not found. Please try another name.")

st.markdown("---")
st.markdown("*Data updated through current season | Predictions with 95% confidence intervals*")
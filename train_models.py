import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import joblib

# Load preprocessed data
data = pd.read_csv('preprocessed_data_advanced.csv')

# Corrected feature set (matches preprocessed CSV)
features = [
    'PTS_WINSOR', 'REB_WINSOR', 'AST_WINSOR',
    'DEF_FG%', 'DEF_3P%', 'TS%', 'MIN'
]

targets = ['NEXT_PTS', 'NEXT_REB', 'NEXT_AST']

def train_model(target):
    X = data[features]
    y = data[target]
    
    model = Pipeline([
        ('regressor', GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    model.fit(X, y)
    joblib.dump(model, f'advanced_model_{target}.pkl')
    print(f"✅ Trained model for {target}")

# Train models
for target in targets:
    train_model(target)
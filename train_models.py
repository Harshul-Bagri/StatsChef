import pandas as pd
import numpy as np
from mapie.regression import MapieRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
# Trains the models using MapieRegressor and GradientBoostingRegressing. It uses a TimeSeriesSplit to evaluate the models and calculate the mean squared error, the mean absolute error and the R^2 score to evaluate the correctness and accuracy of the mode.
def train_final_models():
    data = pd.read_csv('preprocessed_data.csv')
    
    features = [
        'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'FG_PCT', 'FG3_PCT', 'FT_PCT', 'EFF',
        'EMA_PTS', 'EMA_REB', 'EMA_AST',
        'ROLLING_MED_PTS', 'ROLLING_MED_REB', 'ROLLING_MED_AST',
        'LAST_5_AVG_PTS'
    ]
    
    targets = ['NEXT_PTS', 'NEXT_REB', 'NEXT_AST']
    
    best_params = {
        'NEXT_PTS': {
            'n_estimators': 200,
            'learning_rate': 0.15,
            'max_depth': 5,
            'min_samples_leaf': 20,
            'max_features': 0.8
        },
        'NEXT_REB': {
            'n_estimators': 180,
            'learning_rate': 0.18,
            'max_depth': 4,
            'min_samples_leaf': 35
        },
        'NEXT_AST': {
            'n_estimators': 150,
            'learning_rate': 0.12,
            'max_depth': 4,
            'min_samples_leaf': 30
        }
    }
    
    models = {}
    
    for target in targets:
        print(f"\n=== Training {target} ===")
        
        # Adjusted Mapie configuration
        model = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', MapieRegressor(
                estimator=GradientBoostingRegressor(
                    loss='huber',
                    random_state=42,
                    **best_params[target]
                ),
                method="plus",  # Changed to "plus"
                cv=TimeSeriesSplit(n_splits=5, gap=10),  # Adjusted n_splits and gap
                agg_function="mean"  # Changed to mean
            ))
        ])
        
        X = data[features]
        y = data[target]
        
        # Check for NaN values
        if X.isnull().any().any() or y.isnull().any():
            print(f"Warning: NaN values found in {target}. Cleaning data...")
            X = X.fillna(X.median())  # Fill NaN with median
            y = y.fillna(y.median())
        
        # Modified cross-validation
        tscv = TimeSeriesSplit(n_splits=5, gap=10)
        scores = cross_val_score(
            model, X, y,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            error_score='raise'
        )
        print(f"Cross-val MAE: {-scores.mean():.2f} ± {scores.std():.2f}")
        
        # Final training
        model.fit(X, y)
        models[target] = model
        
        # Correct prediction extraction
        preds = model.predict(X)
        print(f"MAE: {mean_absolute_error(y, preds):.2f}")
        print(f"R²: {r2_score(y, preds):.2f}")
    
    joblib.dump(models, 'final_models.pkl')
    print("\n✅ Models saved")

if __name__ == '__main__':
    train_final_models()
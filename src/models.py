from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from typing import Dict, Any


class ModelFactory:
    @staticmethod
    def get_model(name: str, **kwargs) -> Any:
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42,
                **kwargs
            ),
            'xgboost': XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **kwargs
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                **kwargs
            ),
            'ridge': Ridge(alpha=1.0, **kwargs)
        }
        if name not in models:
            raise ValueError(f"Model {name} not available. Choose from: {list(models.keys())}")
        return models[name]
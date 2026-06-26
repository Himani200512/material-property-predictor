import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt

from data_loader import MaterialDataLoader
from features import MaterialFeatureEngineer

def main():
    dataset_name = 'band_gap'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"Training RANDOM FOREST on {dataset_name}")
    print("=" * 60)

    loader = MaterialDataLoader(data_dir='data')
    df = loader.load(dataset_name)
    print(f"Loaded {len(df)} samples")

    feature_engineer = MaterialFeatureEngineer()
    X, y, feature_names = feature_engineer.fit_transform(df)
    print(f"Features: {len(feature_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    metrics = {
        'model': 'random_forest',
        'dataset': dataset_name,
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'test_r2': float(r2_score(y_test, y_pred_test)),
    }

    print("\n--- Random Forest Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred_test, alpha=0.3, s=8, c='steelblue')
    min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Random Forest: {dataset_name}\nMAE={metrics["test_mae"]:.3f}, R²={metrics["test_r2"]:.3f}')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/parity_random_forest_{dataset_name}.png', dpi=150)
    print(f"\nSaved parity plot")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importances[indices], color='steelblue')
    plt.yticks(range(20), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Features: Random Forest on {dataset_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/importance_random_forest_{dataset_name}.png', dpi=150)
    print(f"Saved feature importance plot")

    joblib.dump(model, f'{results_dir}/model_random_forest_{dataset_name}.joblib')
    joblib.dump(feature_engineer, f'{results_dir}/feature_engineer_{dataset_name}.joblib')
    with open(f'{results_dir}/metrics_random_forest_{dataset_name}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model and metrics")


if __name__ == "__main__":
    main()
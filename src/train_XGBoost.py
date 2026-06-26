import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import MaterialDataLoader
from features import MaterialFeatureEngineer
from models import ModelFactory


class Trainer:

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_engineer = MaterialFeatureEngineer()
        self.results_dir = config.get('results_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self, dataset_name: str, model_name: str = 'random_forest'):
        print(f"\n{'='*50}")
        print(f"Training {model_name} on {dataset_name}")
        print(f"{'='*50}\n")

        # 1. Load data
        loader = MaterialDataLoader(self.config.get('data_dir', 'data'))
        df = loader.load(dataset_name)

        # 2. Feature engineering
        print("Engineering features...")
        if 'composition' in df.columns:
            X, y, feature_names = self.feature_engineer.fit_transform_composition(
                df, 'composition'
            )
        else:
            from pymatgen.core import Structure
            df['composition'] = df['structure'].apply(
                lambda s: str(s.composition) if isinstance(s, Structure) else str(s)
            )
            X, y, feature_names = self.feature_engineer.fit_transform_composition(
                df, 'composition'
            )

        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. Train model
        print(f"Training {model_name}...")
        self.model = ModelFactory.get_model(model_name)
        self.model.fit(X_train, y_train)

        metrics = self._evaluate(X_train, X_test, y_train, y_test)
        self._save_artifacts(model_name, dataset_name, metrics, feature_names)

        if hasattr(self.model, 'feature_importances_'):
            self._plot_importance(feature_names, model_name, dataset_name)

        return metrics

    def _evaluate(self, X_train, X_test, y_train, y_test) -> dict:
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
        }

        print("\n--- Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred_test, alpha=0.5, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='Perfect')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Parity Plot: {self.config.get("dataset", "material")}')
        ax.legend()

        plot_path = os.path.join(self.results_dir, 'parity_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved parity plot to {plot_path}")

        return metrics

    def _save_artifacts(self, model_name, dataset_name, metrics, feature_names):
        model_path = os.path.join(self.results_dir, f'{model_name}_{dataset_name}.joblib')
        joblib.dump(self.model, model_path)

        metrics_path = os.path.join(self.results_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(self.results_dir, 'features.json'), 'w') as f:
            json.dump(feature_names, f, indent=2)

        print(f"Saved model to {model_path}")

    def _plot_importance(self, feature_names, model_name, dataset_name, top_n=20):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features: {model_name} on {dataset_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        path = os.path.join(self.results_dir, 'feature_importance.png')
        plt.savefig(path, dpi=150)
        print(f"Saved feature importance to {path}")


if __name__ == "__main__":
    config = {
        'data_dir': 'data',
        'results_dir': 'results'
    }

    trainer = Trainer(config)
    metrics = trainer.run('band_gap', model_name='xgboost')

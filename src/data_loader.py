import os
import pandas as pd
from matminer.datasets import load_dataset
from typing import Tuple, Optional


class MaterialDataLoader:

    AVAILABLE_DATASETS = {
        'band_gap': 'matbench_mp_gap',           # 106,113 samples
        'formation_energy': 'matbench_mp_e_form', # 132,752 samples
        'bulk_modulus': 'matbench_log_kvrh',      # 10,987 samples
        'shear_modulus': 'matbench_log_gvrh',     # 10,987 samples
        'dielectric': 'matbench_dielectric',      # 4,764 samples
        'is_metal': 'matbench_mp_is_metal',       # 106,113 samples (classification)
        'steels': 'matbench_steels',              # 312 samples (composition only)
    }

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load(self, dataset_name: str, save_local: bool = True) -> pd.DataFrame:
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                             f"Available: {list(self.AVAILABLE_DATASETS.keys())}")

        matbench_name = self.AVAILABLE_DATASETS[dataset_name]
        print(f"Loading {matbench_name} from Matminer...")

        df = load_dataset(matbench_name)

        if save_local:
            path = os.path.join(self.data_dir, f"{dataset_name}.csv")
            df_save = df.copy()
            if 'structure' in df_save.columns:
                df_save['structure'] = df_save['structure'].apply(lambda x: str(x))
            df_save.to_csv(path, index=False)
            print(f"Saved to {path}")

        print(f"Loaded {len(df)} samples")
        return df

    def load_local(self, dataset_name: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self.data_dir, f"{dataset_name}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None


if __name__ == "__main__":
    loader = MaterialDataLoader()
    df = loader.load('band_gap')
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Target stats:\n{df['gap pbe'].describe()}")
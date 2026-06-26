import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from matminer.featurizers.conversions import StrToComposition
from sklearn.preprocessing import StandardScaler
from typing import Union, List


class MaterialFeatureEngineer:

    def __init__(self):
        self.composition_converter = StrToComposition()
        self.element_featurizer = ElementProperty.from_preset("magpie")
        self.stoich_featurizer = Stoichiometry()
        self.scaler = StandardScaler()
        self._fitted = False
        self.feature_cols = None

    def fit_transform_composition(self, df: pd.DataFrame,
                                  composition_col: str = 'composition') -> tuple:
        df = df.copy()

        temp_col = f"{composition_col}_str"
        df = df.rename(columns={composition_col: temp_col})
        df = self.composition_converter.featurize_dataframe(df, temp_col)
        df = self.element_featurizer.featurize_dataframe(df, 'composition')
        df = self.stoich_featurizer.featurize_dataframe(df, 'composition')

        target_candidates = ['gap pbe', 'e_form', 'log10(K_VRH)',
                             'log10(G_VRH)', 'is_metal', 'n', 'target',
                             'yield strength', 'tensile strength', 'elongation']

        exclude_cols = {composition_col, temp_col, 'structure', 'composition'} | set(target_candidates)

        self.feature_cols = [c for c in df.columns
                             if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

        X = df[self.feature_cols].values
        y = self._extract_target(df)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True

        return pd.DataFrame(X_scaled, columns=self.feature_cols), y, self.feature_cols

    def transform_composition(self, df: pd.DataFrame,
                              composition_col: str = 'composition') -> np.ndarray:
        if not self._fitted:
            raise ValueError("Must call fit_transform_composition first!")

        df = df.copy()
        temp_col = f"{composition_col}_str"
        df = df.rename(columns={composition_col: temp_col})

        df = self.composition_converter.featurize_dataframe(df, temp_col)
        df = self.element_featurizer.featurize_dataframe(df, 'composition')
        df = self.stoich_featurizer.featurize_dataframe(df, 'composition')

        X = df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X)

    def _extract_target(self, df: pd.DataFrame) -> pd.Series:
        target_candidates = ['gap pbe', 'e_form', 'log10(K_VRH)',
                             'log10(G_VRH)', 'is_metal', 'n', 'target',
                             'yield strength', 'tensile strength', 'elongation']
        for col in target_candidates:
            if col in df.columns:
                return df[col]
        raise ValueError("No target column found in dataframe. Columns: " + str(df.columns.tolist()))


if __name__ == "__main__":
    from data_loader import MaterialDataLoader

    loader = MaterialDataLoader()
    df = loader.load('steels')

    engineer = MaterialFeatureEngineer()
    X, y, features = engineer.fit_transform_composition(df, 'composition')

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sample features: {features[:5]}")
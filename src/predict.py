import joblib
import pandas as pd
from features import MaterialFeatureEngineer

model = joblib.load('results/random_forest_steels.joblib')

# New material to predict
new_material = pd.DataFrame({
    'composition': ['Fe0.6C0.4', 'Al2O3', 'Ti6Al4V']
})

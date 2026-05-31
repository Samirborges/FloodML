from src.database.load_data import LoadData
from src.preprocessing.feature_engineering import FeatureEngineering
from src.models.random_forest import RandomForestModel

from sklearn.model_selection import cross_val_score

df = LoadData.load_meteorological_data()

X, y = FeatureEngineering.build_dataset()

model = RandomForestModel.get_model()

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("F1 Scores:", scores)
print("Mean F1:", scores.mean())
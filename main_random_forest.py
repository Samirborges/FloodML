from src.models.random_forest import RandomForestModel
from src.evaluation.evaluete import EvaluateModel
from src.preprocessing.split_data import SplitData
from src.evaluation.feature_importance import FeatureImportance


X_train, X_test, y_train, y_test = SplitData.split()

model = RandomForestModel.train(X_train, y_train)

EvaluateModel.evaluate(model, X_test, y_test)
FeatureImportance.plot(model, X_train)
from src.preprocessing.split_data import SplitData
from src.models.mlp_model import MLPModel
from src.evaluation.evaluete import EvaluateModel

X_train, X_test, y_train, y_test = SplitData.split()

model = MLPModel.train(X_train, y_train)

EvaluateModel.evaluate(model, X_test, y_test)
from src.models.baseline_logistic import LogisticModel
from src.preprocessing.split_data import SplitData

X_train, X_test, y_train, y_test = SplitData.split()

model = LogisticModel.train(X_train, y_train)

print(model)
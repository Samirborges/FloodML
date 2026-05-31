import pandas as pd


class ErrorAnalysis:

    @staticmethod
    def analyze(model, X_test, y_test):

        predictions = model.predict(X_test)

        errors = X_test[predictions != y_test]

        errors = errors.copy()

        errors["real"] = y_test[predictions != y_test]
        errors["predicted"] = predictions[predictions != y_test]

        print(errors.head())

        return errors
from sklearn.model_selection import cross_val_score
import numpy as np


class CrossValidation:

    @staticmethod
    def validate(model, X, y):

        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="f1"
        )

        print("Scores:", scores)
        print("F1 Médio:", np.mean(scores))
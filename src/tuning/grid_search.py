from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class RandomForestTuning:

    @staticmethod
    def tune(X_train, y_train):

        params = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            params,
            cv=3,
            scoring="f1"
        )

        grid.fit(X_train, y_train)

        print(grid.best_params_)
        print(grid.best_score_)

        return grid.best_estimator_
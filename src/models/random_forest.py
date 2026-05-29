from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:

    @staticmethod
    def train(X_train, y_train):

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        return model
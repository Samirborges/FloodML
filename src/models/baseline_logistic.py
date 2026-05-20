from sklearn.linear_model import LogisticRegression


class LogisticModel:

    @staticmethod
    def train(X_train, y_train):

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train, y_train)

        return model
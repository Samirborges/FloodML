from sklearn.neural_network import MLPClassifier


class MLPModel:

    @staticmethod
    def train(X_train, y_train):

        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42
        )

        model.fit(X_train, y_train)

        return model
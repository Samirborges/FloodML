from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:

    @staticmethod
    def get_model():
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )

    @staticmethod
    def train(X_train, y_train):
        model = RandomForestModel.get_model()
        model.fit(X_train, y_train)
        return model
from sklearn.model_selection import train_test_split
from src.preprocessing.feature_engineering import FeatureEngineering


class SplitData:

    @staticmethod
    def split():

        X, y = FeatureEngineering.build_dataset()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test
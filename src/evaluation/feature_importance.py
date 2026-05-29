import pandas as pd
import matplotlib.pyplot as plt


class FeatureImportance:

    @staticmethod
    def plot(model, X_train):

        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        })

        importance = importance.sort_values(
            by="importance",
            ascending=False
        )

        print("\nFeature Importance:\n")
        print(importance)

        importance.plot(
            x="feature",
            y="importance",
            kind="bar",
            legend=False
        )

        plt.title("Importância das Features")
        plt.ylabel("Importância")
        plt.xlabel("Variáveis")

        plt.tight_layout()
        plt.show()
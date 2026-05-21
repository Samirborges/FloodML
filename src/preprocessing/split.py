# src/preprocessing/split.py
from sklearn.model_selection import train_test_split

def dividir_dados(X_estudo, y_gabarito, test_size=0.3, random_state=42):
    """Divide as matrizes em conjuntos de treino e teste."""
    print("A separar dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_estudo, y_gabarito, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
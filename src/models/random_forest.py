# src/models/random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def treinar_modelo_rf(X_train, y_train, X_test, y_test):
    """Treina o modelo Random Forest e simula eventos climáticos extremos."""
    print("[SP03/05] 🧠 A treinar IA preditiva e a simular tempestade extrema...")
    
    # Pipeline inicial
    pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))])
    otimizador = GridSearchCV(pipeline, {'rf__n_estimators': [50], 'rf__max_depth': [10, None]}, cv=3)
    otimizador.fit(X_train, y_train)
    melhor_pipeline = otimizador.best_estimator_

    # Simulação de extremos climáticos (Warm Start)
    modelo_rf_continuo = melhor_pipeline.named_steps['rf']
    modelo_rf_continuo.set_params(warm_start=True)
    X_sintetico, y_sintetico = X_test.copy(), y_test.copy()
    
    idx_extremos = np.random.choice(len(y_sintetico), size=int(0.2 * len(y_sintetico)), replace=False)
    X_sintetico[idx_extremos, 1] *= 5.0
    y_sintetico[idx_extremos] = 1
    
    modelo_rf_continuo.n_estimators += 20  
    modelo_rf_continuo.fit(melhor_pipeline.named_steps['scaler'].transform(X_sintetico), y_sintetico)

    return melhor_pipeline
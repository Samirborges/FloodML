# src/models/evaluation.py
from sklearn.metrics import accuracy_score, classification_report

def avaliar_modelo(melhor_pipeline, X_test, y_test):
    """Calcula e exibe as métricas técnicas do modelo treinado."""
    y_pred_test = melhor_pipeline.predict(X_test)
    
    print("\n" + "-"*60)
    print("📊 MÉTRICAS TÉCNICAS INTEGRAIS DE VALIDAÇÃO DA IA:")
    print(f"Acurácia Geral do Modelo: {accuracy_score(y_test, y_pred_test) * 100:.2f}%")
    print("-" * 60)
    print(classification_report(y_test, y_pred_test, target_names=['Terreno Seguro', 'Área de Risco'], zero_division=0))
    print("-" * 60)
    
    return y_pred_test
# src/visualization/metrics_plots.py
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plotar_matriz_confusao(y_test, y_pred_test):
    """Gera e salva o gráfico da Matriz de Confusão."""
    fig, ax = plt.subplots(figsize=(5, 4)) 
    ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred_test, 
        display_labels=['Terreno Seguro', 'Área de Risco'], 
        cmap='Blues', 
        ax=ax
    )
    plt.title("Matriz de Confusão Estatística")
    
    # Salva a imagem na pasta correta
    caminho_salvar = "results/figures/matriz_confusao.png"
    plt.savefig(caminho_salvar)
    plt.show()
    print(f"✅ Gráfico salvo em: {caminho_salvar}")
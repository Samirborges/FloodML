# main.py
import argparse
from src.database.supabase_loader import inicializar_earth_engine, descarregar_dados_geograficos
from src.preprocessing.feature_engineering import gerar_features
from src.preprocessing.split import dividir_dados
from src.models.random_forest import treinar_modelo_rf
from src.models.evaluation import avaliar_modelo
from src.visualization.plots import calcular_rota_fuga, gerar_mapa_interativo
from src.visualization.metrics_plots import plotar_matriz_confusao

def executar_pipeline_completo(coords):
    """Executa o fluxo completo do modelo de inundação sem intervenção manual."""
    try:
        inicializar_earth_engine()
        
        # 1. Carregar
        relevo, agua, estruturas, _, limites = descarregar_dados_geograficos(coords)
        
        # 2. Processar
        X_estudo, y_gabarito, feature_relevo, fluxo = gerar_features(relevo, agua)
        X_train, X_test, y_train, y_test = dividir_dados(X_estudo, y_gabarito)
        
        # 3. Treinar e Avaliar
        pipeline = treinar_modelo_rf(X_train, y_train, X_test, y_test)
        y_pred = avaliar_modelo(pipeline, X_test, y_test)
        plotar_matriz_confusao(y_test, y_pred)
        
        # 4. Gerar Rotas e Mapa
        rota_coords, mapa_previsao = calcular_rota_fuga(pipeline, X_estudo, relevo, estruturas, feature_relevo, limites)
        gerar_mapa_interativo(limites, relevo, agua, fluxo, mapa_previsao, estruturas, rota_coords)
        
        print("\n✅ Processo concluído com sucesso! Verifique a pasta 'results/figures/'.")
        
    except Exception as e:
        print(f"❌ Erro na execução: {e}")

if __name__ == "__main__":
    # Coordenadas fixas de exemplo (um quadrado genérico) para teste automatizado
    coordenadas_exemplo = [
        [-47.95, -15.80],
        [-47.90, -15.80],
        [-47.90, -15.75],
        [-47.95, -15.75],
        [-47.95, -15.80]
    ]
    
    print("A iniciar o sistema em modo automatizado (CLI)...")
    executar_pipeline_completo(coordenadas_exemplo)
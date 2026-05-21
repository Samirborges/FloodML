# src/visualization/plots.py
import numpy as np
import heapq
import folium
import matplotlib.pyplot as plt

def calcular_rota_fuga(melhor_pipeline, X_estudo, relevo_original, estruturas_original, feature_relevo, limites_geograficos):
    """Mapeia a malha urbana e calcula a rota de fuga anticolisão (Algoritmo Dijkstra)."""
    lat_min, lat_max, lon_min, lon_max = limites_geograficos
    mapa_previsao = melhor_pipeline.predict_proba(X_estudo)[:, 1].reshape(relevo_original.shape)
    linhas, colunas = mapa_previsao.shape
    
    start = (linhas // 2, colunas // 2)
    bordas = [(r, 0) for r in range(linhas)] + [(r, colunas-1) for r in range(linhas)] + \
             [(0, c) for c in range(colunas)] + [(linhas-1, c) for c in range(colunas)]
    end = min(bordas, key=lambda p: mapa_previsao[p] - feature_relevo[p])
    
    # Penalização para evitar construções
    penalidade_casas = np.where(estruturas_original == 50, 500.0, 0.0)
    custo_movimento = (mapa_previsao * 100.0) + penalidade_casas + 1.0

    distancias = np.full((linhas, colunas), np.inf)
    distancias[start] = custo_movimento[start]
    fila_prioridade = [(custo_movimento[start], start)]
    came_from = {}
    direcoes = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    while fila_prioridade:
        custo_atual, (r, c) = heapq.heappop(fila_prioridade)
        if (r, c) == end: break
        if custo_atual > distancias[r, c]: continue
        
        for dr, dc in direcoes:
            nr, nc = r + dr, c + dc
            if 0 <= nr < linhas and 0 <= nc < colunas:
                peso_diag = 1.414 if dr != 0 and dc != 0 else 1.0
                novo_custo = custo_atual + (custo_movimento[nr, nc] * peso_diag)
                if novo_custo < distancias[nr, nc]:
                    distancias[nr, nc] = novo_custo
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(fila_prioridade, (novo_custo, (nr, nc)))
    
    rota_indices, atual = [], end
    while atual in came_from:
        rota_indices.append(atual)
        atual = came_from[atual]
    rota_indices.append(start)
    rota_indices.reverse()
    
    rota_coords = [[lat_max - (r / (linhas - 1)) * (lat_max - lat_min), 
                    lon_min + (c / (colunas - 1)) * (lon_max - lon_min)] for r, c in rota_indices]
    
    return rota_coords, mapa_previsao

def gerar_mapa_interativo(limites_geograficos, relevo_original, agua_original, fluxo_agua_realista, mapa_previsao, estruturas_original, rota_coords):
    """Compila o mapa interativo com todas as camadas geográficas."""
    print("🖥️ A compilar mapa interativo com todas as camadas geográficas...")
    lat_min, lat_max, lon_min, lon_max = limites_geograficos
    
    mapa_resultados = folium.Map(location=[(lat_min + lat_max)/2, (lon_min + lon_max)/2], zoom_start=14, tiles="CartoDB dark_matter")

    # Salvar matrizes temporárias
    plt.imsave('layer_historico.png', agua_original.reshape(relevo_original.shape), cmap='Blues')
    plt.imsave('layer_fluxo.png', fluxo_agua_realista, cmap='YlGnBu')
    plt.imsave('layer_relevo.png', relevo_original, cmap='terrain')
    plt.imsave('layer_ia.png', mapa_previsao, cmap='coolwarm')
    mask_casas = np.where(estruturas_original == 50, 1, np.nan)
    plt.imsave('layer_casas.png', mask_casas, cmap='autumn') 

    # Inserção das camadas no Folium
    folium.raster_layers.ImageOverlay(image='layer_relevo.png', bounds=[[lat_min, lon_min], [lat_max, lon_max]], opacity=0.5, name="Relevo Topográfico (SRTM)", show=False).add_to(mapa_resultados)
    folium.raster_layers.ImageOverlay(image='layer_historico.png', bounds=[[lat_min, lon_min], [lat_max, lon_max]], opacity=0.6, name="Histórico de Água (JRC)", show=False).add_to(mapa_resultados)
    folium.raster_layers.ImageOverlay(image='layer_fluxo.png', bounds=[[lat_min, lon_min], [lat_max, lon_max]], opacity=0.6, name="Fluxo Hídrico Simulador", show=False).add_to(mapa_resultados)
    folium.raster_layers.ImageOverlay(image='layer_ia.png', bounds=[[lat_min, lon_min], [lat_max, lon_max]], opacity=0.6, name="Previsão de Risco Final (IA)", show=True).add_to(mapa_resultados)
    folium.raster_layers.ImageOverlay(image='layer_casas.png', bounds=[[lat_min, lon_min], [lat_max, lon_max]], opacity=0.5, name="Obstáculos (Áreas Construídas)", show=True).add_to(mapa_resultados)
    
    # Rota
    folium.Marker(rota_coords[0], popup="Ponto de Partida", icon=folium.Icon(color="blue", icon="info-sign")).add_to(mapa_resultados)
    folium.Marker(rota_coords[-1], popup="Zona Segura (Borda Alta)", icon=folium.Icon(color="green", icon="ok-sign")).add_to(mapa_resultados)
    folium.PolyLine(rota_coords, color="lime", weight=5, opacity=0.9, name="Rota de Fuga Segura (Dijkstra)").add_to(mapa_resultados)
    
    folium.LayerControl(collapsed=False).add_to(mapa_resultados)
    
    arquivo_html = "results/figures/mapa_da_sua_regiao.html"
    mapa_resultados.save(arquivo_html)
    print(f"✅ MAPA GERADO! Salvo em: '{arquivo_html}'")
    
    return mapa_resultados
# src/database/supabase_loader.py
import ee
import numpy as np
import geemap

def inicializar_earth_engine():
    """Inicializa a conexão com a API do Google Earth Engine."""
    try:
        ee.Initialize(project='banded-advice-416822')
        print("Conexão com Earth Engine estabelecida.")
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='banded-advice-416822')

def descarregar_dados_geograficos(coords):
    """Descarrega relevo, água histórica e mapeamento de estruturas (ESA)."""
    print("[SP02] 📡 A descarregar Relevo, Água Histórica e Mapeamento de Estruturas (ESA)...")
    
    # Extrair limites
    lon_min, lon_max = min([p[0] for p in coords]), max([p[0] for p in coords])
    lat_min, lat_max = min([p[1] for p in coords]), max([p[1] for p in coords])
    regiao = ee.Geometry.BBox(lon_min, lat_min, lon_max, lat_max)
    
    # Importar Datasets
    dataset_relevo = ee.Image('USGS/SRTMGL1_003').rename('elevacao')
    dataset_agua = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').rename('agua')
    dataset_estruturas = ee.ImageCollection("ESA/WorldCover/v200").first().select('Map').rename('landcover')
    
    # Combinar e converter para numpy
    imagem_combinada = ee.Image.cat([dataset_relevo, dataset_agua, dataset_estruturas]).clip(regiao)
    dados_numpy = geemap.ee_to_numpy(imagem_combinada, region=regiao, scale=30)
    
    relevo_original = np.nan_to_num(dados_numpy[:, :, 0], nan=0)
    agua_original = np.nan_to_num(dados_numpy[:, :, 1], nan=0)
    estruturas_original = np.nan_to_num(dados_numpy[:, :, 2], nan=0) # Classe 50 = Área Construída
    
    limites_geograficos = (lat_min, lat_max, lon_min, lon_max)
    
    return relevo_original, agua_original, estruturas_original, regiao, limites_geograficos
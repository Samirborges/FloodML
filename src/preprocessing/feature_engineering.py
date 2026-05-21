# src/preprocessing/feature_engineering.py
import numpy as np
from scipy.ndimage import gaussian_filter

def gerar_features(relevo_original, agua_original):
    """Gera matrizes de estudo a partir da simulação de fluxo e relevo."""
    print("[SP04] 🌊 A simular dinâmica de escoamento e declividade...")
    
    max_elev = np.max(relevo_original) if np.max(relevo_original) > 0 else 1
    feature_relevo = relevo_original / max_elev
    
    # Target (gabarito)
    y_gabarito = np.where(agua_original > 50, 1, 0).flatten()

    # Cálculo de gradiente e declividade
    dy, dx = np.gradient(relevo_original)
    declividade = np.sqrt(dx**2 + dy**2)
    norm_declividade = declividade / (np.max(declividade) if np.max(declividade) > 0 else 1)
    
    # Simulação de fluxo
    fluxo_bruto = (1.0 - feature_relevo) * (1.0 - norm_declividade) * 2.5
    fluxo_agua_realista = gaussian_filter(fluxo_bruto, sigma=0.5) 
    
    # Compilar matriz X de estudo
    X_estudo = np.column_stack((feature_relevo.flatten(), fluxo_agua_realista.flatten()))
    
    return X_estudo, y_gabarito, feature_relevo, fluxo_agua_realista
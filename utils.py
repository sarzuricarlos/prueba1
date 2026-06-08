import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import torch
import random

# CONFIGURACIÓN GLOBAL

def get_device():
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("ℹ️  GPU no disponible, usando CPU")
    return device

def set_seed(seed=42):
    """
    Fija la semilla aleatoria para reproducibilidad
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Semilla fijada a {seed}")

# PROCESAMIENTO DE DATOS

def calcular_variables_derivadas(df):

    df = df.copy()

    # 1. Diferencia de tiempo de pantalla en fin de semana
    df['screen_time_weekend_difference'] = df['weekend_screen_time'] - df['daily_screen_time_hours']

    # 2-4. Ratios de uso (respecto al tiempo total de pantalla)
    df['social_media_ratio'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + 0.01)
    df['gaming_ratio'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + 0.01)
    df['work_study_ratio'] = df['work_study_hours'] / (df['daily_screen_time_hours'] + 0.01)

    # 5. Índice de intensidad de uso
    df['usage_intensity_index'] = (df['notifications_per_day'] + df['app_opens_per_day']) / 100

    # 6. Ratio de productividad (trabajo/estudio vs entretenimiento)
    df['productivity_ratio'] = df['work_study_hours'] / (df['social_media_hours'] + df['gaming_hours'] + 0.01)

    # 7. Intensidad de notificaciones (por hora de pantalla)
    df['notification_intensity'] = df['notifications_per_day'] / (df['daily_screen_time_hours'] + 0.01)

    # 8. Índice compuesto de uso (ponderado)
    df['total_usage_composite'] = (
        df['daily_screen_time_hours'] * 0.3 +
        df['social_media_hours'] * 0.25 +
        df['gaming_hours'] * 0.15 +
        df['work_study_hours'] * 0.1 +
        (df['notifications_per_day'] / 100) * 0.2
    )

    # 9. Desbalance sueño-pantalla
    df['sleep_screen_imbalance'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 0.01)
    
    # 10. Dominancia de entretenimiento
    df['entertainment_dominance'] = (df['social_media_hours'] + df['gaming_hours']) / (df['work_study_hours'] + 0.01)
    
    # 11. Compulsividad digital
    df['digital_compulsivity'] = df['app_opens_per_day'] / (df['daily_screen_time_hours'] + 0.01)

    return df

def cargar_artefactos(artifacts_dir):
    """
    Carga los artefactos del proyecto (scaler, label_encoder, feature_columns)
    
    Args:
        artifacts_dir (Path o str): Directorio de artefactos
    
    Returns:
        tuple: (scaler, label_encoder, feature_columns, class_mapping, class_names)
    """
    artifacts_dir = Path(artifacts_dir)
    
    # Cargar scaler
    with open(artifacts_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Cargar label encoder
    with open(artifacts_dir / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Cargar feature columns
    with open(artifacts_dir / 'feature_columns.json', 'r') as f:
        feature_info = json.load(f)
        feature_columns = feature_info['feature_columns']
        class_mapping = feature_info['class_mapping']
        class_names = list(class_mapping.keys())
    
    print(f"✅ Artefactos cargados:")
    print(f"   - Features: {len(feature_columns)} variables")
    print(f"   - Clases: {class_names}")
    
    return scaler, label_encoder, feature_columns, class_mapping, class_names

# ============================================
# UTILIDADES GENERALES
# ============================================

def get_project_root():
    return Path(__file__).parent

def get_artifacts_path():
    return get_project_root() / 'datasets' / 'artifacts'


def cargar_class_names():
    """Carga los nombres de clases desde feature_columns.json"""
    artifacts_path = get_artifacts_path()
    with open(artifacts_path / 'feature_columns.json', 'r') as f:
        feature_info = json.load(f)
    return list(feature_info['class_mapping'].keys())

if __name__ == "__main__":
    get_device()
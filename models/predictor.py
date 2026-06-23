import numpy as np
import pandas as pd
import json
import pickle
import os
import sys
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / 'datasets' / 'limpio_procesado' / 'dataset_limpio.csv'
ARTIFACTS_DIR = ROOT_DIR / 'datasets' / 'artifacts'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints' / 'predictor'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Importar utils desde la raíz
sys.path.append(str(ROOT_DIR))
from utils import get_device, set_seed, cargar_class_names


# FUNCIÓN PARA CALCULAR EL RIESGO COMPUESTO

def calcular_riesgo_compuesto(df):

    df = df.copy()

    # Funciones de riesgo
    def riesgo_pantalla(x):
        return np.clip((x - 6) / 6, 0, 1)

    def riesgo_sueño(x):
        return np.clip((8 - x) / 3, 0, 1)

    def riesgo_redes(x):
        return np.clip((x - 1.5) / 4, 0, 1)

    def riesgo_nocturno(x):
        return np.clip((x - 1) / 5, 0, 1)

    def riesgo_notificaciones(x):
        return np.clip((x - 80) / 180, 0, 1)

    # Columnas requeridas
    columnas_riesgo = {
        'pantalla': ('daily_screen_time_hours_raw', 'daily_screen_time_hours'),
        'sueño': ('sleep_hours_raw', 'sleep_hours'),
        'redes': ('social_media_hours_raw', 'social_media_hours'),
        'notificaciones': ('notifications_per_day_raw', 'notifications_per_day'),
        'nocturno': ('screen_time_weekend_difference_raw', 'screen_time_weekend_difference')
    }

    # Detectar disponibilidad de _raw
    usar_raw = all(col_raw in df.columns for col_raw, _ in columnas_riesgo.values())
    if not usar_raw:
        print("⚠️  Columnas _raw no encontradas, usando fallback a columnas normales (riesgo incorrecto)")

    factores = {}
    for nombre, (col_raw, col_normal) in columnas_riesgo.items():
        if usar_raw and col_raw in df.columns:
            valores = df[col_raw].values
        else:
            valores = df[col_normal].values

        if nombre == 'pantalla':
            factores[nombre] = riesgo_pantalla(valores)
        elif nombre == 'sueño':
            factores[nombre] = riesgo_sueño(valores)
        elif nombre == 'redes':
            factores[nombre] = riesgo_redes(valores)
        elif nombre == 'nocturno':
            factores[nombre] = riesgo_nocturno(valores)
        elif nombre == 'notificaciones':
            factores[nombre] = riesgo_notificaciones(valores)

    pesos = {
        'pantalla': 0.25,
        'sueño': 0.20,
        'redes': 0.20,
        'nocturno': 0.25,
        'notificaciones': 0.10,
    }
    riesgo = sum(factores[k] * pesos[k] for k in pesos.keys())
   
    riesgo = np.clip(riesgo, 0, 1)
    return riesgo

# CLASE PRINCIPAL: PREDICTOR XGBOOST

class PredictorXGBoost:

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.history = None
        self.is_trained = False

        # Hiperparámetros XGBoost
        self.params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }

    def entrenar(self, X_train, y_train, X_val, y_val, verbose=False):

        print("\n" + "=" * 60)
        print("🧠 ENTRENANDO PREDICTOR XGBOOST")
        print("=" * 60)

        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        print(f"\n📊 Configuración de entrenamiento:")
        print(f"   - Muestras train: {len(X_train)}")
        print(f"   - Muestras val: {len(X_val)}")
        print(f"   - n_estimators: {self.params['n_estimators']}")
        print(f"   - max_depth: {self.params['max_depth']}")
        print(f"   - learning_rate: {self.params['learning_rate']}")
        print("\n" + "-" * 40)

        self.model = xgb.XGBRegressor(**self.params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=verbose
        )

        # Guardar historial de pérdida
        self.history = {
            'train_loss': self.model.evals_result()['validation_0']['rmse'],
            'val_loss': self.model.evals_result()['validation_1']['rmse']
        }
        self.is_trained = True

        print(f"\n✅ Entrenamiento completado. Mejor RMSE en val: {min(self.history['val_loss']):.4f}")
        return self

    def predecir(self, X):

        if not self.is_trained or self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado o cargado.")
        X = np.array(X, dtype=np.float32)
        return self.model.predict(X)

    def inferir(self, X):

        riesgo = self.predecir(X)
        riesgo = float(riesgo[0]) if isinstance(riesgo, np.ndarray) else float(riesgo)
        # Categorizar
        if riesgo < 0.33:
            categoria = "Bajo"
        elif riesgo < 0.66:
            categoria = "Medio"
        else:
            categoria = "Alto"
        return riesgo, categoria

    def get_feature_importance(self):

        if not self.is_trained or self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado o cargado.")
        return self.model.feature_importances_

    def guardar_checkpoint(self, directorio=None):

        if directorio is None:
            directorio = CHECKPOINT_DIR
        directorio = Path(directorio)
        directorio.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        # Guardar modelo en formato JSON
        model_path = directorio / 'modelo_xgboost.json'
        self.model.save_model(str(model_path))
        print(f"✅ Modelo XGBoost guardado en: {model_path}")

        # Metadatos
        metadata = {
            'feature_names': self.feature_names,
            'history': self.history,
            'params': self.params,
            'is_trained': self.is_trained,
            'version': 'xgboost_v1'
        }
        metadata_path = directorio / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadatos guardados en: {metadata_path}")

        if self.history:
            hist_df = pd.DataFrame(self.history)
            hist_df.to_csv(directorio / 'historial_xgboost.csv', index=False)
            print(f"✅ Historial guardado en: {directorio / 'historial_xgboost.csv'}")

    def cargar_checkpoint(self, directorio=None):

        if directorio is None:
            directorio = CHECKPOINT_DIR
        directorio = Path(directorio)

        model_path = directorio / 'modelo_xgboost.json'
        metadata_path = directorio / 'metadata.pkl'

        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"No se encontraron metadatos en: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.feature_names = metadata['feature_names']
        self.history = metadata['history']
        self.params = metadata['params']
        self.is_trained = metadata['is_trained']

        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(str(model_path))

        print(f"✅ Modelo XGBoost cargado desde: {directorio}")
        print(f"   - Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
        return self

# FUNCIONES AUXILIARES

def cargar_datos():

    print("=" * 60)
    print("📂 CARGANDO DATOS")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

    with open(ARTIFACTS_DIR / 'feature_columns.json', 'r') as f:
        feature_info = json.load(f)

    feature_columns = feature_info['feature_columns']
    print(f"✅ Features: {len(feature_columns)} variables")

    # Calcular riesgo compuesto
    print("\n📊 Calculando riesgo compuesto (target)...")
    y = calcular_riesgo_compuesto(df)
    print(f"   Min: {y.min():.3f}, Max: {y.max():.3f}, Media: {y.mean():.3f}")

    X = df[feature_columns].values.astype(np.float32)
    return X, y, feature_columns

def preparar_datos(X, y, test_size=0.2, val_size=0.2, random_state=42):

    print("\n" + "=" * 60)
    print("📊 PREPARANDO DATOS")
    print("=" * 60)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )

    print(f"✅ Entrenamiento: {X_train.shape[0]} registros")
    print(f"✅ Validación: {X_val.shape[0]} registros")
    print(f"✅ Prueba: {X_test.shape[0]} registros")

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluar_modelo(predictor, X_test, y_test, feature_names=None):

    print("\n" + "=" * 60)
    print("📊 EVALUANDO PREDICTOR XGBOOST")
    print("=" * 60)

    y_pred = predictor.predecir(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Métricas de rendimiento:")
    print(f"   ✅ MSE: {mse:.4f}")
    print(f"   ✅ RMSE: {rmse:.4f}")
    print(f"   ✅ MAE: {mae:.4f}")
    print(f"   ✅ R²: {r2:.4f}")

    # Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(y_test, y_pred, alpha=0.3)
    ax1.plot([0, 1], [0, 1], 'r--', label='Predicción perfecta')
    ax1.set_xlabel('Riesgo real')
    ax1.set_ylabel('Riesgo predicho')
    ax1.set_title('Predicciones vs Reales')
    ax1.legend()
    ax1.grid(True)

    errores = y_pred - y_test
    ax2.hist(errores, bins=30, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', label='Error cero')
    ax2.set_xlabel('Error de predicción')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de errores')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / 'evaluacion_xgboost.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfica guardada en: {CHECKPOINT_DIR / 'evaluacion_xgboost.png'}")

    importancias = predictor.get_feature_importance()
    if importancias is not None and len(importancias) > 0:
        indices = np.argsort(importancias)[::-1][:5]
        
        if feature_names is None:
            feature_names = predictor.feature_names
        
        print("\n📊 Top 5 características más importantes (XGBoost):")
        for i, idx in enumerate(indices):
            nombre = feature_names[idx] if feature_names and idx < len(feature_names) else f"Feature_{idx}"
            print(f"   {i+1}. {nombre}: {importancias[idx]:.4f}")
    
    return mse, rmse, mae, r2, y_pred

def guardar_graficas_entrenamiento(history):

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train_loss'], label='Train RMSE')
    ax.plot(history['val_loss'], label='Val RMSE')
    ax.set_title('RMSE durante el entrenamiento')
    ax.set_xlabel('Época')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / 'historial_xgboost.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráficas guardadas en: {CHECKPOINT_DIR / 'historial_xgboost.png'}")

def main():
    device = get_device()
    set_seed(42)
    print("=" * 60)
    print("🚀 PREDICTOR XGBOOST - RADAR DIGITAL")
    print("=" * 60)
    print("\n🎯 Características:")
    print("   ✅ Modelo basado en árboles (XGBoost)")
    print("   ✅ Interpretabilidad nativa (importancia de features)")
    print("   ✅ Entrenamiento rápido y robusto")
    print("   ✅ Early stopping con paciencia 50")
    print("   ✅ Métricas: RMSE, MAE, R²\n")

    # 1. Cargar datos y calcular target
    X, y, feature_columns = cargar_datos()
    class_names = cargar_class_names()

    # 2. Dividir
    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos(X, y)

    # 3. Entrenar
    predictor = PredictorXGBoost()
    predictor.feature_names = feature_columns
    predictor.entrenar(X_train, y_train, X_val, y_val, verbose=False)

    # 4. Guardar checkpoint
    predictor.guardar_checkpoint()

    # 5. Evaluar
    mse, rmse, mae, r2, y_pred = evaluar_modelo(predictor, X_test, y_test, feature_names=feature_columns)

    # 6. Guardar gráficas de entrenamiento
    if predictor.history:
        guardar_graficas_entrenamiento(predictor.history)

    print("\n" + "=" * 60)
    print("✅ PREDICTOR XGBOOST ENTRENADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\n📊 Resumen final:")
    print(f"   R²: {r2:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Checkpoint: {CHECKPOINT_DIR}")

    return predictor, r2

if __name__ == "__main__":
    main()

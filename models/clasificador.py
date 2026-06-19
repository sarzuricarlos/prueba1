import torch
import numpy as np
import pandas as pd
import json
import pickle
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / 'datasets' / 'limpio_procesado' / 'dataset_limpio.csv'
ARTIFACTS_DIR = ROOT_DIR / 'datasets' / 'artifacts'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints' / 'clasificador'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Importar utils desde la raíz
sys.path.append(str(ROOT_DIR))
from utils import get_device, set_seed, cargar_class_names

class ClasificadorTabNet:
    def __init__(self, device='auto'):
        
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device

        self.model = None
        self.class_names = None
        self.feature_names = None
        self.input_size = None
        self.history = None
        self.is_trained = False

        self.tabnet_params = {
            'n_d': 8,
            'n_a': 8,
            'n_steps': 5,
            'gamma': 2.5,
            'lambda_sparse': 1e-3,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=5e-3),
            'mask_type': 'sparsemax',
            'device_name': self.device,
            'verbose': 1
        }

    def _calcular_val_loss(self, X_val, y_val):
        if self.model is None:
            return None
        probas = self.model.predict_proba(X_val)
        epsilon = 1e-12
        probas = np.clip(probas, epsilon, 1. - epsilon)
        loss = -np.mean(np.log(probas[np.arange(len(y_val)), y_val]))
        return loss

    def entrenar(self, X_train, y_train, X_val, y_val,
                 epochs=300, batch_size=128, patience=30,
                 virtual_batch_size=64):
        print("\n" + "=" * 60)
        print("🧠 ENTRENANDO CLASIFICADOR TABNET")
        print("=" * 60)

        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)
        y_val = np.array(y_val, dtype=np.int64)

        self.input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))

        # Balanceo de clases
        print("\n⚖️ Calculando pesos de clase para balanceo...")
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = np.array([class_weights[i] for i in y_train])
        for i, w in enumerate(class_weights):
            print(f"      Clase {i}: {w:.3f}")

        print("\n🏗️ Construyendo modelo TabNet...")
        self.model = TabNetClassifier(**self.tabnet_params)

        print(f"\n📊 Configuración de entrenamiento:")
        print(f"   - Épocas máximas: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Virtual batch size: {virtual_batch_size}")
        print(f"   - Paciencia: {patience}")
        print(f"   - Clases: {num_classes}")
        print(f"   - Muestras train: {len(X_train)}")
        print(f"   - Muestras val: {len(X_val)}")
        print("\n" + "-" * 40)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['val'],
            eval_metric=['accuracy'],
            max_epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            weights=sample_weights,
            num_workers=0,
            drop_last=False
        )

        # Construir historial desde el diccionario interno
        history_dict = self.model.history.history
        history = {
            'train_loss': history_dict['loss'],
            'val_accuracy': history_dict['val_accuracy']
        }
        if 'val_loss' in history_dict:
            history['val_loss'] = history_dict['val_loss']
        else:
            history['val_loss'] = self._calcular_val_loss(X_val, y_val)

        self.history = history
        self.is_trained = True

        print(f"\n✅ Entrenamiento completado. Mejor accuracy en val: {max(history['val_accuracy']):.4f}")
        return self

    # Métodos de inferencia

    def predecir(self, X):
        if not self.is_trained or self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado o cargado.")
        X = np.array(X, dtype=np.float32)
        return self.model.predict(X)

    def predecir_proba(self, X):
        if not self.is_trained or self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado o cargado.")
        X = np.array(X, dtype=np.float32)
        return self.model.predict_proba(X)

    def inferir(self, X):
        probs = self.predecir_proba(X)
        clase_idx = int(np.argmax(probs, axis=1)[0])
        nombre_clase = self.class_names[clase_idx] if self.class_names else str(clase_idx)
        return clase_idx, nombre_clase, probs[0].tolist()

    def get_feature_importance(self):
        if not self.is_trained or self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado o cargado.")
        return self.model.feature_importances_

    # Guardado y carga

    def guardar_checkpoint(self, directorio=None):
        if directorio is None:
            directorio = CHECKPOINT_DIR
        directorio = Path(directorio)
        directorio.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        params_to_save = self.tabnet_params.copy()
        if 'device_name' in params_to_save:
            params_to_save['device_name'] = str(params_to_save['device_name'])

        params_to_save.pop('optimizer_fn', None)
        params_to_save.pop('optimizer_params', None)

        # Guardar modelo TabNet
        model_path = directorio / 'modelo_tabnet'
        self.model.save_model(str(model_path))
        print(f"✅ Modelo TabNet guardado en: {model_path}")

        # Guardar metadatos
        metadata = {
            'class_names': self.class_names,
            'feature_names': self.feature_names,
            'input_size': self.input_size,
            'history': self.history,
            'tabnet_params': params_to_save,
            'original_device': self.device,
            'is_trained': self.is_trained,
            'version': 'tabnet_v1'
        }
        metadata_path = directorio / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadatos guardados en: {metadata_path}")

        if self.history:
            hist_df = pd.DataFrame(self.history)
            hist_df.to_csv(directorio / 'historial_tabnet.csv', index=False)
            print(f"✅ Historial guardado en: {directorio / 'historial_tabnet.csv'}")

    def cargar_checkpoint(self, directorio=None):
        if directorio is None:
            directorio = CHECKPOINT_DIR
        directorio = Path(directorio)

        model_path = directorio / 'modelo_tabnet.zip'
        metadata_path = directorio / 'metadata.pkl'

        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"No se encontraron metadatos en: {metadata_path}")

        # Cargar metadatos
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.class_names = metadata['class_names']
        self.feature_names = metadata['feature_names']
        self.input_size = metadata['input_size']
        self.history = metadata['history']
        self.is_trained = metadata['is_trained']
        self.device = metadata.get('original_device', 'auto')
        params = metadata.get('tabnet_params', {})

        # Reconstruir parámetros completos
        params['optimizer_fn'] = torch.optim.Adam
        params['optimizer_params'] = dict(lr=2e-2)
        params['device_name'] = self.device
        params['verbose'] = 1

        self.tabnet_params = params

        # Inicializar y cargar modelo
        self.model = TabNetClassifier(**self.tabnet_params)
        self.model.load_model(str(model_path))

        print(f"✅ Modelo TabNet cargado desde: {directorio}")
        print(f"   - Clases: {self.class_names}")
        print(f"   - Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
        print(f"   - Versión: {metadata.get('version', 'desconocida')}")
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
    target_column = feature_info['target_column']
    class_names = cargar_class_names()
    print(f"✅ Features: {len(feature_columns)} variables")
    print(f"✅ Target: {target_column}")
    print(f"✅ Clases: {class_names}")
    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(np.int64)
    return X, y, feature_columns, class_names

def preparar_datos(X, y, test_size=0.2, val_size=0.2, random_state=42):
    print("\n" + "=" * 60)
    print("📊 PREPARANDO DATOS")
    print("=" * 60)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    print(f"✅ Entrenamiento: {X_train.shape[0]} registros")
    print(f"✅ Validación: {X_val.shape[0]} registros")
    print(f"✅ Prueba: {X_test.shape[0]} registros")
    print("\n📊 Distribución de clases en entrenamiento:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   Clase {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluar_modelo(clasificador, X_test, y_test, class_names=None):
    print("\n" + "=" * 60)
    print("📊 EVALUANDO CLASIFICADOR TABNET")
    print("=" * 60)
    if class_names is None:
        class_names = ['None', 'Mild', 'Moderate', 'Severe']
    y_pred = clasificador.predecir(X_test)
    y_proba = clasificador.predecir_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy en prueba: {acc:.4f}")
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Matriz de Confusión (Absoluta)')
    axes[0].set_xlabel('Predicción')
    axes[0].set_ylabel('Real')
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Matriz de Confusión (Normalizada)')
    axes[1].set_xlabel('Predicción')
    axes[1].set_ylabel('Real')
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / 'confusion_matrix_tabnet.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Matriz de confusión guardada en: {CHECKPOINT_DIR / 'confusion_matrix_tabnet.png'}")
    if clasificador.is_trained:
        importancias = clasificador.get_feature_importance()
        if importancias is not None and len(importancias) > 0:
            print("\n📊 Top 5 características más importantes (TabNet):")
            indices = np.argsort(importancias)[::-1][:5]
            for i, idx in enumerate(indices):
                nombre = clasificador.feature_names[idx] if clasificador.feature_names else f"Feature_{idx}"
                print(f"   {i+1}. {nombre}: {importancias[idx]:.4f}")
    return acc, y_pred, y_proba

def guardar_graficas_entrenamiento(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss'] is not None:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history['val_accuracy'], label='Val Accuracy', color='green')
    ax2.set_title('Precisión en validación')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / 'historial_tabnet.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráficas guardadas en: {CHECKPOINT_DIR / 'historial_tabnet.png'}")

def main():
    device = get_device()
    set_seed(42)
    print("=" * 60)
    print("🚀 CLASIFICADOR TABNET - RADAR DIGITAL")
    print("=" * 60)
    device_str = str(device)
    print(f"📱 Dispositivo: {device_str}")
    print("\n🎯 Características de TabNet:")
    print("   ✅ Arquitectura basada en atención sparsemax")
    print("   ✅ Interpretabilidad mediante máscaras de features")
    print("   ✅ Balanceo de clases con sample weights")
    print("   ✅ Early stopping con paciencia")
    print("   ✅ Selección automática de características")
    print("   ✅ Métricas: Accuracy, Precision, Recall, F1\n")
    X, y, feature_columns, class_names = cargar_datos()
    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos(X, y)
    clasificador = ClasificadorTabNet(device=device_str)
    clasificador.class_names = class_names
    clasificador.feature_names = feature_columns
    clasificador.entrenar(
        X_train, y_train, X_val, y_val,
        epochs=500,
        batch_size=128,
        patience=100,
        virtual_batch_size=128
    )
    clasificador.guardar_checkpoint()
    acc, y_pred, y_proba = evaluar_modelo(clasificador, X_test, y_test, class_names)
    if clasificador.history:
        guardar_graficas_entrenamiento(clasificador.history)
    print("\n" + "=" * 60)
    print("✅ CLASIFICADOR TABNET ENTRENADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\n📊 Resumen final:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Checkpoint: {CHECKPOINT_DIR}")
    return clasificador, acc

if __name__ == "__main__":
    main()

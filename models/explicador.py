import numpy as np
import pandas as pd
import json
import os
import sys
import shap
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / 'datasets' / 'limpio_procesado' / 'dataset_limpio.csv'
ARTIFACTS_DIR = ROOT_DIR / 'datasets' / 'artifacts'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints' / 'explicador'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

sys.path.append(str(ROOT_DIR))
from utils import get_device, set_seed, cargar_class_names

class ExplicadorMLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(ExplicadorMLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.LogSoftmax(dim=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# CLASE PRINCIPAL: EXPLICADOR

class Explicador:

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None          # MLP aproximador
        self.feature_names = None
        self.history = None
        self.is_trained = False

    def generar_etiquetas_shap(self, X, y, n_samples=2000, n_estimators=200, max_depth=10):

        print("\n" + "=" * 60)
        print("🔍 GENERANDO ETIQUETAS SHAP CON RANDOM FOREST")
        print("=" * 60)
        print(f"📊 Muestras a procesar: {min(n_samples, len(X))}")

        # Muestra aleatoria
        indices = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        n_samples, n_features = X_sample.shape

        # Entrenar Random Forest
        print("⏳ Entrenando Random Forest para SHAP...")
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_sample, y_sample)

        # Crear explainer TreeExplainer
        explainer = shap.TreeExplainer(rf)
        print("⏳ Calculando SHAP values...")
        shap_values = explainer.shap_values(X_sample)

        # Inicializar matriz de importancias absolutas
        shap_abs = np.zeros((n_samples, n_features))

        # Caso 1: shap_values es una lista
        if isinstance(shap_values, list):
            for class_shap in shap_values:
                # Asegurar que cada elemento sea 2D
                if class_shap.ndim == 3:
                    # Si es 3D (n_samples, n_features, n_classes) -> promediar sobre última dim
                    class_shap = class_shap.mean(axis=2)
                elif class_shap.ndim != 2:
                    raise ValueError(f"Forma inesperada en class_shap: {class_shap.shape}")
                shap_abs += np.abs(class_shap)
            shap_abs = shap_abs / len(shap_values)  # Promedio sobre clases

        # Caso 2: shap_values es un array único
        else:
            if shap_values.ndim == 3:
                # (n_samples, n_features, n_classes) -> promediar sobre clases
                shap_abs = np.abs(shap_values).mean(axis=2)
            elif shap_values.ndim == 2:
                shap_abs = np.abs(shap_values)
            else:
                raise ValueError(f"Forma inesperada de shap_values: {shap_values.shape}")

        # Normalizar por muestra (suma = 1 por fila)
        row_sums = shap_abs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10  # Evitar división por cero
        shap_norm = shap_abs / row_sums

        print(f"✅ SHAP values generados: {shap_norm.shape}")

        # --- Visualización ---
        importancias_promedio = shap_norm.mean(axis=0)
        indices_orden = np.argsort(importancias_promedio)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(n_features), importancias_promedio[indices_orden])
        if self.feature_names:
            plt.xticks(range(n_features),
                    [self.feature_names[i] for i in indices_orden],
                    rotation=45, ha='right')
        plt.title('Importancia Promedio de Características (SHAP)')
        plt.tight_layout()
        plt.savefig(CHECKPOINT_DIR / 'shap_importancias.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Gráfico guardado en: {CHECKPOINT_DIR / 'shap_importancias.png'}")

        # Top 5
        print("\n📊 Top 5 características más importantes (SHAP):")
        for i, idx in enumerate(indices_orden[:5]):
            nombre = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
            print(f"   {i+1}. {nombre}: {importancias_promedio[idx]:.4f}")

        return shap_norm, X_sample

    def entrenar(self, X_train, y_train, X_val, y_val,
                 input_size, output_size,
                 epochs=300, batch_size=64, learning_rate=0.001, patience=20):
        """
        Entrena el MLP aproximador con KLDivLoss.
        """
        print("\n" + "=" * 60)
        print("🧠 ENTRENANDO EXPLICADOR MLP")
        print("=" * 60)
        print(f"📱 Dispositivo: {self.device}")

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.model = ExplicadorMLP(input_size, output_size).to(self.device)
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"\n📊 Configuración:")
        print(f"   - Épocas: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}")
        print("\n" + "-" * 40)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_preds.extend(torch.exp(outputs).cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            val_mae = mean_absolute_error(val_targets, val_preds)
            val_r2 = r2_score(val_targets, val_preds)

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            scheduler.step(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Época {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {history['train_loss'][-1]:.4f} | "
                      f"Val Loss: {history['val_loss'][-1]:.4f} | "
                      f"MAE: {val_mae:.3f} | R²: {val_r2:.3f}")

            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping en época {epoch+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Mejor modelo restaurado (Val Loss: {best_val_loss:.4f})")

        self.history = history
        self.is_trained = True
        return self

    def predecir_importancias(self, X):
        """
        Predice importancias para nuevos datos (sin SHAP).
        Retorna: array de importancias (n_samples, n_features).
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado o cargado.")
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            log_probs = self.model(X_t)
            importancias = torch.exp(log_probs).cpu().numpy()
        return importancias

    def inferir(self, X):
        """
        Método unificado para app.py.
        Retorna: ranking de (feature, importancia) ordenado descendente.
        """
        importancias = self.predecir_importancias(X)[0]  # primera muestra
        ranking = [(self.feature_names[i], importancias[i]) for i in range(len(importancias))]
        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        return ranking

    def guardar_checkpoint(self, directorio=None):
        if directorio is None:
            directorio = CHECKPOINT_DIR
        directorio = Path(directorio)
        directorio.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        model_path = directorio / 'modelo_explicador.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.network[0].in_features,
            'output_size': self.model.network[-2].out_features,
            'feature_names': self.feature_names,
            'history': self.history,
            'is_trained': self.is_trained,
            'version': 'explicador_mlp_v1'
        }, model_path)
        print(f"✅ Modelo explicador guardado en: {model_path}")

        if self.history:
            hist_df = pd.DataFrame(self.history)
            hist_df.to_csv(directorio / 'historial_explicador.csv', index=False)
            print(f"✅ Historial guardado en: {directorio / 'historial_explicador.csv'}")

    def cargar_checkpoint(self, directorio=None):
        if directorio is None:
            directorio = CHECKPOINT_DIR
        directorio = Path(directorio)

        model_path = directorio / 'modelo_explicador.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.feature_names = checkpoint['feature_names']
        self.history = checkpoint['history']
        self.is_trained = checkpoint['is_trained']

        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size']
        self.model = ExplicadorMLP(input_size, output_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✅ Modelo explicador cargado desde: {directorio}")
        print(f"   - Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
        return self


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
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
    print(f"✅ Features: {len(feature_columns)} variables")
    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(np.int64)
    return X, y, feature_columns

def preparar_datos(X, y, test_size=0.2, val_size=0.2, random_state=42):
    print("\n" + "=" * 60)
    print("📊 PREPARANDO DATOS")
    print("=" * 60)
    # Sin stratify porque y es continuo (importancias SHAP)
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

def evaluar_modelo(explicador, X_test, y_test):
    print("\n" + "=" * 60)
    print("📊 EVALUANDO EXPLICADOR MLP")
    print("=" * 60)
    y_pred = explicador.predecir_importancias(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 Métricas de rendimiento:")
    print(f"   ✅ MAE: {mae:.4f}")
    print(f"   ✅ R²: {r2:.4f}")
    return mae, r2

def guardar_graficas_entrenamiento(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Pérdida durante entrenamiento')
    ax1.set_xlabel('Época')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history['val_mae'], label='MAE', color='green')
    ax2.set_title('Error Absoluto Medio (MAE)')
    ax2.set_xlabel('Época')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / 'historial_explicador.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráficas guardadas en: {CHECKPOINT_DIR / 'historial_explicador.png'}")

def main():
    device = get_device()
    set_seed(42)
    print("=" * 60)
    print("🚀 EXPLICADOR (SHAP + MLP) - RADAR DIGITAL")
    print("=" * 60)
    print(f"📱 Dispositivo: {device}")
    print("\n🎯 Características:")
    print("   ✅ Usa SHAP con Random Forest (TreeExplainer)")
    print("   ✅ Entrena MLP para aproximar importancias")
    print("   ✅ Interpretabilidad mediante ranking de features")
    print("   ✅ Métricas: MAE, R²\n")

    # 1. Cargar datos y feature names
    X, y, feature_names = cargar_datos()

    # 2. Generar etiquetas SHAP
    explicador = Explicador(device=device)
    explicador.feature_names = feature_names
    y_shap, X_sample = explicador.generar_etiquetas_shap(X, y, n_samples=2000)

    # 3. Dividir en train/val/test
    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos(X_sample, y_shap)

    # 4. Entrenar MLP aproximador
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    explicador.entrenar(
        X_train, y_train, X_val, y_val,
        input_size=input_size,
        output_size=output_size,
        epochs=300,
        batch_size=64,
        learning_rate=0.001,
        patience=20
    )

    # 5. Guardar y evaluar
    explicador.guardar_checkpoint()
    mae, r2 = evaluar_modelo(explicador, X_test, y_test)
    guardar_graficas_entrenamiento(explicador.history)

    print("\n" + "=" * 60)
    print("✅ EXPLICADOR ENTRENADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\n📊 Resumen final:")
    print(f"   R²: {r2:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   Checkpoint: {CHECKPOINT_DIR}")

    return explicador, r2

if __name__ == "__main__":
    main()
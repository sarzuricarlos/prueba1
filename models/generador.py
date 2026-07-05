import numpy as np
import pandas as pd
import json
import pickle
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / 'datasets' / 'limpio_procesado' / 'dataset_limpio.csv'
ARTIFACTS_DIR = ROOT_DIR / 'datasets' / 'artifacts'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints' / 'generador'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

sys.path.append(str(ROOT_DIR))
from utils import get_device, set_seed, cargar_class_names

# Importar modelos entrenados
from models.clasificador import ClasificadorTabNet
from models.predictor import PredictorXGBoost
from models.explicador import Explicador

# BASE DE CONOCIMIENTO

BASE_CONOCIMIENTO = {
    "daily_screen_time_hours": {
        "bajo": [
            "Tu tiempo de pantalla ({valor:.1f}h) es adecuado, sigue así.",
            "El tiempo que pasas en el teléfono ({valor:.1f}h) es razonable, mantén este hábito.",
        ],
        "medio": [
            "Tu tiempo de pantalla ({valor:.1f}h) es aceptable, pero intenta reducirlo un poco más.",
            "Considera usar aplicaciones que monitoreen y limiten tu tiempo de pantalla.",
        ],
        "alto": [
            "Reduce tu tiempo de pantalla ({valor:.1f}h) a menos de 6 horas diarias.",
            "Establece horarios libres de teléfono, especialmente durante comidas y antes de dormir.",
            "Reemplaza parte del tiempo de pantalla con actividades offline como leer o caminar.",
        ],
    },
    "social_media_hours": {
        "bajo": [
            "Tu uso de redes sociales ({valor:.1f}h) es saludable, continúa así.",
        ],
        "medio": [
            "Tu uso de redes sociales ({valor:.1f}h) es moderado; intenta bajarlo de 2 horas diarias.",
            "Considera desactivar las notificaciones push de redes sociales.",
        ],
        "alto": [
            "Reduce tu tiempo en redes sociales ({valor:.1f}h) a menos de 2 horas diarias.",
            "Prueba a eliminar las apps de redes sociales y usar solo la versión web.",
            "Realiza un 'detox digital' periódico de redes sociales.",
        ],
    },
    "sleep_hours": {
        "bajo": [
            "Duerme al menos 7 horas ({valor:.1f}h actuales) para recuperar energía.",
            "Evita el teléfono al menos 1 hora antes de dormir.",
        ],
        "medio": [
            "Tu sueño ({valor:.1f}h) es aceptable, pero 7-8 horas es lo ideal.",
        ],
        "alto": [
            "Mantén tus buenos hábitos de sueño ({valor:.1f}h).",
        ],
    },
    "night_usage": {
        "bajo": [
            "Tu uso nocturno del teléfono es bajo, excelente.",
        ],
        "medio": [
            "Disminuye el uso nocturno del teléfono para mejorar tu descanso.",
            "Activa el modo nocturno para reducir la exposición a luz azul.",
        ],
        "alto": [
            "Apaga el teléfono 1 hora antes de dormir para reducir tu uso nocturno.",
            "Deja el teléfono en otra habitación durante la noche.",
        ],
    },
    "notifications_per_day": {
        "bajo": [
            "Mantén un control saludable de tus notificaciones.",
        ],
        "medio": [
            "Organiza tus notificaciones para revisarlas en momentos específicos.",
        ],
        "alto": [
            "Desactiva las notificaciones de las apps que no son esenciales.",
            "Agrupa las notificaciones para revisarlas en momentos específicos.",
        ],
    },
}

REGLAS_NIVEL = {
    "daily_screen_time_hours": {"alto": 8, "medio": 4},
    "social_media_hours": {"alto": 5, "medio": 2.5},
    "notifications_per_day": {"alto": 200, "medio": 100},
}


def determinar_nivel(feature, valor, datos_usuario=None):
    if feature == "sleep_hours":
        if valor < 6:
            return "bajo"
        elif valor < 7:
            return "medio"
        return "alto"
    if feature == "night_usage":

        if valor > 1.5:
            return "alto"
        elif valor > 0.8:
            return "medio"
        return "bajo"
    reglas = REGLAS_NIVEL.get(feature)
    if reglas is None:
        return "medio"
    if valor > reglas["alto"]:
        return "alto"
    elif valor > reglas["medio"]:
        return "medio"
    return "bajo"


def _texto_para_variable(feature, valor, datos_usuario=None):
    if feature not in BASE_CONOCIMIENTO:
        return None
    nivel = determinar_nivel(feature, valor, datos_usuario)
    opciones = BASE_CONOCIMIENTO[feature].get(nivel)
    if not opciones:
        return None
    plantilla = np.random.choice(opciones)
    return plantilla.format(valor=valor)


def _completar_valores_derivados(raw_values):

    raw_values = dict(raw_values)

    if 'night_usage_raw' not in raw_values:
        screen = raw_values.get('daily_screen_time_hours_raw')
        sleep = raw_values.get('sleep_hours_raw')
        if screen is not None and sleep is not None:
            raw_values['night_usage_raw'] = screen / (sleep + 1)

    if 'sleep_hours' not in raw_values and 'sleep_hours_raw' in raw_values:
        raw_values['sleep_hours'] = raw_values['sleep_hours_raw']

    return raw_values


def generar_texto_recomendacion(clase, urgencia, ranking, raw_values):
    """
    Genera texto usando valores crudos (raw_values) en lugar de normalizados.
    """
    raw_values = _completar_valores_derivados(raw_values)
    lineas = []
    if urgencia >= 0.66:
        intro = "⚠️ Tu patrón de uso digital indica un nivel de urgencia alto. Es importante tomar acción pronto:"
    elif urgencia >= 0.33:
        intro = "📊 Tus hábitos digitales requieren atención. Considera estas recomendaciones:"
    else:
        intro = "✅ Tienes hábitos digitales saludables. Sigue estas recomendaciones para mantenerlos:"
    lineas.append(intro)
    lineas.append("")

    for feature, _ in ranking[:3]:
        # Buscar el valor crudo correspondiente
        raw_col = f"{feature}_raw"
        if raw_col in raw_values:
            valor = raw_values[raw_col]
        elif feature in raw_values:
            valor = raw_values[feature]
        else:
            valor = 0.0  # fallback

        texto = _texto_para_variable(feature, valor, raw_values)
        if texto:
            lineas.append(f"• {texto}")

    lineas.append("")
    if clase == "Severe":
        lineas.append("💡 Considera buscar ayuda profesional para manejar tu relación con la tecnología.")
    elif clase == "Moderate":
        lineas.append("💡 Pequeños cambios diarios pueden marcar una gran diferencia en tus hábitos digitales.")
    else:
        lineas.append("💡 Mantén estos buenos hábitos y revisa tu progreso regularmente.")
    return "\n".join(lineas)


class GeneradorMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout_rate=0.3):
        super(GeneradorMLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# CLASE PRINCIPAL: GENERADOR

class Generador:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.feature_names = None
        self.class_names = None
        self.raw_columns = None
        self.history = None
        self.is_trained = False

    def generar_corpus(self, clasificador, predictor, explicador,
                   X, y, df, feature_names, class_names, n_samples=3000):

        print("\n" + "=" * 60)
        print("📚 GENERANDO CORPUS SINTÉTICO")
        print("=" * 60)
        print(f"📊 Muestras a procesar: {min(n_samples, len(X))}")

        # Identificar columnas raw
        raw_cols = [col for col in df.columns if col.endswith('_raw')]
        print(f"   Columnas raw disponibles: {raw_cols}")

        indices = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
        X_sample = X[indices]
        y_sample = y[indices]

        corpus = []
        for i in range(len(X_sample)):
            idx = indices[i]
            x = X_sample[i:i+1]
            
            # Obtener valores crudos del DataFrame
            raw_values = {col: float(df.iloc[idx][col]) for col in raw_cols}

            clase_idx, clase_nombre, _ = clasificador.inferir(x)
            riesgo, _ = predictor.inferir(x)
            ranking = explicador.inferir(x)

            clase_normalizada = clase_idx / (len(class_names) - 1)
            urgencia = 0.5 * clase_normalizada + 0.5 * riesgo

            features_corpus = X_sample[i]  # SOLO features originales

            texto = generar_texto_recomendacion(clase_nombre, urgencia, ranking, raw_values)

            corpus.append({
                "features": features_corpus,  # <-- SOLO features originales
                "urgencia": urgencia,         # <-- target
                "raw_values": raw_values,
                "texto": texto
            })

        corpus_df = pd.DataFrame(corpus)
        corpus_path = CHECKPOINT_DIR / 'corpus_sintetico.csv'
        corpus_df.to_csv(corpus_path, index=False)
        print(f"✅ Corpus guardado en: {corpus_path}")
        print(f"   - Ejemplos: {len(corpus)}")
        print(f"   - Features por muestra: {features_corpus.shape[0]} (solo features originales)")
        return corpus
    
    def entrenar(self, corpus, epochs=150, batch_size=64, learning_rate=0.001, patience=20):
        print("\n" + "=" * 60)
        print("🧠 ENTRENANDO GENERADOR MLP")
        print("=" * 60)
        print(f"📱 Dispositivo: {self.device}")

        X_corpus = np.array([item["features"] for item in corpus], dtype=np.float32)
        y_corpus = np.array([item["urgencia"] for item in corpus], dtype=np.float32)

        X_train, X_temp, y_train, y_temp = train_test_split(X_corpus, y_corpus, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(f"   - Entrenamiento: {len(X_train)}")
        print(f"   - Validación: {len(X_val)}")
        print(f"   - Prueba: {len(X_test)}")

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.model = GeneradorMLP(X_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_r2": []}
        best_val_loss = float("inf")
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
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            val_mae = mean_absolute_error(val_targets, val_preds)
            val_r2 = r2_score(val_targets, val_preds)

            history["train_loss"].append(train_loss / len(train_loader))
            history["val_loss"].append(val_loss / len(val_loader))
            history["val_mae"].append(val_mae)
            history["val_r2"].append(val_r2)

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

        self._guardar_graficas(history)
        self._evaluar(X_test, y_test)

        return self

    def predecir_urgencia(self, X):
        if not self.is_trained or self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado o cargado.")
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

    def inferir(self, estado, X=None):
        
        clasificacion = estado.get("clasificacion", {})
        prediccion = estado.get("prediccion", {})
        explicacion = estado.get("explicacion", {})
        datos_usuario = estado.get("datos_usuario", {})

        clase = clasificacion.get("nombre", "Moderate")
        riesgo = prediccion.get("riesgo", 0.5)
        ranking = explicacion.get("ranking", [])

        raw_values = datos_usuario

        if not raw_values:
            for feature, _ in ranking:
                raw_values[feature] = 0.0

        # Urgencia: usar MLP entrenado SOLO con features originales
        if self.is_trained and self.model is not None and X is not None:
            # Extraer SOLO las 18 features originales para el MLP
            if X.shape[1] == 20:
                X_features = X[:, :18]  # Tomar solo las primeras 18
                urgencia = float(self.predecir_urgencia(X_features)[0])
            elif X.shape[1] == 18:
                # Compatibilidad hacia atrás
                urgencia = float(self.predecir_urgencia(X)[0])
            else:
                print(f"⚠️  Dimensiones inesperadas: {X.shape[1]}, usando riesgo como fallback")
                urgencia = riesgo
        else:
            urgencia = riesgo

        texto = generar_texto_recomendacion(clase, urgencia, ranking, raw_values)
        return {"texto": texto, "urgencia": urgencia, "prioridad": prediccion.get("categoria", "Medio")}

    def guardar_checkpoint(self, directorio=None):
        if directorio is None:
            directorio = CHECKPOINT_DIR
        directorio = Path(directorio)
        directorio.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        model_path = directorio / "modelo_generador.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_size": self.model.network[0].in_features,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "raw_columns": self.raw_columns,
            "history": self.history,
            "is_trained": self.is_trained,
            "version": "generador_mlp_v2"
        }, model_path)
        print(f"✅ Modelo guardado en: {model_path}")

        if self.history:
            hist_df = pd.DataFrame(self.history)
            hist_df.to_csv(directorio / "historial_generador.csv", index=False)
            print(f"✅ Historial guardado en: {directorio / 'historial_generador.csv'}")

    def cargar_checkpoint(self, directorio=None):
        if directorio is None:
            directorio = CHECKPOINT_DIR
        directorio = Path(directorio)

        model_path = directorio / "modelo_generador.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.feature_names = checkpoint.get("feature_names")
        self.class_names = checkpoint.get("class_names")
        self.raw_columns = checkpoint.get("raw_columns")
        self.history = checkpoint.get("history")
        self.is_trained = checkpoint.get("is_trained", False)

        input_size = checkpoint["input_size"]
        self.model = GeneradorMLP(input_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"✅ Modelo cargado desde: {directorio}")
        return self

    def _guardar_graficas(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_title("Pérdida durante el entrenamiento")
        ax1.set_xlabel("Época")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(history["val_mae"], label="MAE", color="green")
        ax2.set_title("Error Absoluto Medio (MAE)")
        ax2.set_xlabel("Época")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(CHECKPOINT_DIR / "historial_generador.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Gráficas guardadas en: {CHECKPOINT_DIR / 'historial_generador.png'}")

    def _evaluar(self, X_test, y_test):
        print("\n" + "=" * 60)
        print("📊 EVALUANDO GENERADOR MLP")
        print("=" * 60)
        y_pred = self.predecir_urgencia(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"   ✅ MSE: {mse:.4f}")
        print(f"   ✅ RMSE: {rmse:.4f}")
        print(f"   ✅ MAE: {mae:.4f}")
        print(f"   ✅ R²: {r2:.4f}")

# FUNCIÓN PRINCIPAL

def main():
    device = get_device()
    set_seed(42)

    print("=" * 60)
    print("🚀 GENERADOR HÍBRIDO (MLP + REGLAS) - RADAR DIGITAL")
    print("=" * 60)
    print(f"📱 Dispositivo: {device}")
    print("\n🎯 Características:")
    print("   ✅ MLP para estimar urgencia")
    print("   ✅ Base de conocimiento para generar texto")
    print("   ✅ Usa valores crudos (_raw) para recomendaciones")
    print("   ✅ Métricas: MSE, RMSE, MAE, R²\n")

    # 1. Cargar dataset completo (con columnas raw)
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

    with open(ARTIFACTS_DIR / "feature_columns.json", "r") as f:
        feature_info = json.load(f)

    feature_columns = feature_info["feature_columns"]
    target_column = feature_info["target_column"]
    class_names = cargar_class_names()
    raw_columns = feature_info.get("raw_columns_for_risk_score", [])

    print(f"✅ Features: {len(feature_columns)} variables")
    print(f"✅ Columnas raw disponibles: {len(raw_columns)}")
    if raw_columns:
        print(f"   {raw_columns}")

    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(np.int64)

    # 2. Cargar modelos entrenados
    print("\n📂 CARGANDO MODELOS ENTRENADOS")
    print("=" * 60)

    print("⏳ Cargando Clasificador...")
    clasificador = ClasificadorTabNet(device=str(device))
    clasificador.cargar_checkpoint()
    print("✅ Clasificador cargado")

    print("⏳ Cargando Predictor...")
    predictor = PredictorXGBoost()
    predictor.cargar_checkpoint()
    print("✅ Predictor cargado")

    print("⏳ Cargando Explicador...")
    explicador = Explicador(device=device)
    explicador.cargar_checkpoint()
    print("✅ Explicador cargado")

    # 3. Generar corpus
    generador = Generador(device=device)
    generador.feature_names = feature_columns
    generador.class_names = class_names
    generador.raw_columns = raw_columns

    corpus = generador.generar_corpus(
        clasificador, predictor, explicador,
        X, y, df, feature_columns, class_names,
        n_samples=2000
    )

    # 4. Entrenar MLP
    generador.entrenar(corpus, epochs=150, batch_size=64, learning_rate=0.001, patience=20)

    # 5. Guardar checkpoint
    generador.guardar_checkpoint()

    # 6. Probar generación con valores crudos reales
    print("\n" + "=" * 60)
    print("🧪 PROBANDO GENERADOR")
    print("=" * 60)

    # Tomar un usuario aleatorio del DataFrame
    idx = np.random.randint(0, len(df))
    
    # x_test = SOLO las features originales (18 features)
    x_test = X[idx:idx+1]
    
    # Obtener predicciones para construir el estado
    clase_idx, clase_nombre, _ = clasificador.inferir(x_test)
    riesgo, categoria_riesgo = predictor.inferir(x_test)
    ranking = explicador.inferir(x_test)

    # Construir datos_usuario con valores crudos
    datos_usuario = {}
    for col in raw_columns:
        datos_usuario[col] = float(df.iloc[idx][col])
    # También añadir las features sin _raw para compatibilidad
    for col in feature_columns[:5]:
        if col not in datos_usuario:
            datos_usuario[col] = float(df.iloc[idx][col])

    estado_test = {
        "clasificacion": {"nombre": clase_nombre},
        "prediccion": {"riesgo": riesgo, "categoria": categoria_riesgo},
        "explicacion": {"ranking": ranking},
        "datos_usuario": datos_usuario
    }

    print("\n📊 Estado de prueba:")
    print(f"   Clase: {clase_nombre}")
    print(f"   Riesgo: {riesgo:.3f} ({categoria_riesgo})")
    print(f"   Top features: {[f[0] for f in ranking[:3]]}")
    print(f"   Valores crudos:")
    for col in ['daily_screen_time_hours_raw', 'sleep_hours_raw', 'social_media_hours_raw']:
        if col in datos_usuario:
            print(f"      {col}: {datos_usuario[col]:.2f}")

    print("\n📝 Recomendación generada:")
    # PASAR SOLO features originales (SIN clase ni riesgo)
    resultado = generador.inferir(estado_test, X=x_test)
    print(f"   (urgencia estimada por el GeneradorMLP: {resultado['urgencia']:.3f})")
    print(resultado["texto"])

    print("\n" + "=" * 60)
    print("✅ GENERADOR ENTRENADO EXITOSAMENTE")
    print("=" * 60)
    print(f"   Checkpoint: {CHECKPOINT_DIR}")

    r2 = generador.history['val_r2'][-1] if generador.history else 0.0
    return generador, r2 

if __name__ == "__main__":
    main()

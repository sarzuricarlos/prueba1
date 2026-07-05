# app.py - VERSIÓN CORREGIDA (con 18 features finales)

"""
Sistema Radar Digital - Aplicación Principal
Interfaz interactiva por consola para probar el sistema completo
Utiliza los modelos entrenados: TabNet (clasificador), XGBoost (predictor),
MLP SHAP (explicador) y MLP+Reglas (generador).
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Configuración de rutas
ROOT_DIR = Path(__file__).parent
ARTIFACTS_DIR = ROOT_DIR / 'datasets' / 'artifacts'
CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'

# Importar utilidades y modelos
sys.path.append(str(ROOT_DIR))
from utils import (
    get_device, set_seed, cargar_artefactos,
    calcular_variables_derivadas
)
from models.clasificador import ClasificadorTabNet
from models.predictor import PredictorXGBoost
from models.explicador import Explicador
from models.generador import Generador


class RadarDigital:
    """
    Sistema Radar Digital - Clase principal que orquesta los 4 modelos.
    """

    def __init__(self):
        print("=" * 60)
        print("📡 RADAR DIGITAL - Sistema de Análisis de Hábitos Digitales")
        print("=" * 60)
        print("\n⏳ Inicializando sistema...")

        # Configurar dispositivo y semilla
        self.device = str(get_device())  # 'cpu' o 'cuda'
        set_seed(42)

        # Cargar artefactos (scaler, feature_columns, etc.)
        self.artifacts = self.cargar_artefactos()
        self.scaler = self.artifacts['scaler']
        self.feature_columns = self.artifacts['feature_columns']  # 18 columnas
        self.class_names = self.artifacts['class_names']
        self.class_mapping = self.artifacts['class_mapping']

        # Definir el orden completo de 19 columnas que el scaler espera
        # (basado en el notebook actualizado)
        self.scaler_feature_names = [
            'daily_screen_time_hours',
            'social_media_hours',
            'gaming_hours',
            'work_study_hours',
            'sleep_hours',
            'notifications_per_day',
            'app_opens_per_day',
            'weekend_screen_time',          # <--- columna extra (no está en feature_columns)
            'screen_time_weekend_difference',
            'social_media_ratio',
            'gaming_ratio',
            'work_study_ratio',
            'usage_intensity_index',
            'productivity_ratio',
            'notification_intensity',
            'total_usage_composite',
            'sleep_screen_imbalance',       # <--- NUEVA
            'entertainment_dominance',      # <--- NUEVA
            'digital_compulsivity'          # <--- NUEVA
        ]  # Total: 19 columnas

        # Verificación de consistencia
        if hasattr(self.scaler, 'n_features_in_'):
            assert self.scaler.n_features_in_ == len(self.scaler_feature_names), \
                f"El scaler espera {self.scaler.n_features_in_} características, pero se definieron {len(self.scaler_feature_names)}"

        # Inicializar modelos
        self.clasificador = None
        self.predictor = None
        self.explicador = None
        self.generador = None

        # Cargar los modelos entrenados
        self.cargar_modelos()

        print("\n✅ Sistema inicializado correctamente!")
        print("=" * 60)

    def cargar_artefactos(self):
        """
        Carga scaler, label_encoder y feature_columns desde artifacts.
        """
        print("\n📂 Cargando artefactos...")
        scaler, label_encoder, feature_columns, class_mapping, class_names = cargar_artefactos(ARTIFACTS_DIR)

        with open(ARTIFACTS_DIR / 'feature_columns.json', 'r') as f:
            feature_info = json.load(f)
        raw_columns = feature_info.get('raw_columns_for_risk_score', [])

        artifacts = {
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_columns': feature_columns,
            'class_mapping': class_mapping,
            'class_names': class_names,
            'raw_columns': raw_columns
        }
        print(f"   ✅ Features: {len(feature_columns)} variables")
        print(f"   ✅ Clases: {class_names}")
        print(f"   ✅ Columnas raw: {len(raw_columns)}")
        return artifacts

    def cargar_modelos(self):
        """
        Carga los 4 modelos usando sus métodos de carga nativos.
        """
        print("\n📂 Cargando modelos...")

        try:
            self.clasificador = ClasificadorTabNet(device=self.device)
            self.clasificador.cargar_checkpoint(CHECKPOINT_DIR / 'clasificador')
            print("   ✅ Clasificador (TabNet) cargado")
        except Exception as e:
            print(f"   ❌ Error al cargar clasificador: {e}")
            sys.exit(1)

        try:
            self.predictor = PredictorXGBoost()
            self.predictor.cargar_checkpoint(CHECKPOINT_DIR / 'predictor')
            print("   ✅ Predictor (XGBoost) cargado")
        except Exception as e:
            print(f"   ❌ Error al cargar predictor: {e}")
            sys.exit(1)

        try:
            self.explicador = Explicador(device=self.device)
            self.explicador.cargar_checkpoint(CHECKPOINT_DIR / 'explicador')
            self.explicador.feature_names = self.feature_columns
            print("   ✅ Explicador (MLP) cargado")
        except Exception as e:
            print(f"   ❌ Error al cargar explicador: {e}")
            sys.exit(1)

        try:
            self.generador = Generador(device=self.device)
            self.generador.cargar_checkpoint(CHECKPOINT_DIR / 'generador')
            self.generador.feature_names = self.feature_columns
            self.generador.class_names = self.class_names
            self.generador.raw_columns = self.artifacts['raw_columns']
            print("   ✅ Generador (MLP+Reglas) cargado")
        except Exception as e:
            print(f"   ❌ Error al cargar generador: {e}")
            sys.exit(1)

    def preprocesar(self, datos_usuario):
        """
        Convierte datos crudos del usuario en array normalizado de 18 características.
        Internamente construye las 19 columnas que el scaler espera, escala,
        y luego extrae solo las 18 que usan los modelos.
        """
        df = pd.DataFrame([datos_usuario])

        # Calcular variables derivadas (todas las que necesitamos)
        df = calcular_variables_derivadas(df)

        # Asegurar que todas las columnas que espera el scaler estén presentes
        for col in self.scaler_feature_names:
            if col not in df.columns:
                if col == 'weekend_screen_time':
                    df[col] = datos_usuario.get('weekend_screen_time', 0.0)
                else:
                    df[col] = 0.0  # fallback

        # Extraer en el orden exacto que espera el scaler (19 columnas)
        X_full = df[self.scaler_feature_names].values.astype(np.float32)

        # Normalizar con el scaler (entrenado con 19 columnas)
        X_normalized_full = self.scaler.transform(X_full)

        # Extraer solo las 18 columnas que están en self.feature_columns
        indices = [self.scaler_feature_names.index(col) for col in self.feature_columns]
        X_final = X_normalized_full[:, indices]

        return X_final.astype(np.float32)

    def ejecutar(self, datos_usuario):
        """
        Ejecuta el pipeline completo.
        """
        print("\n" + "=" * 60)
        print("🔄 EJECUTANDO PIPELINE")
        print("=" * 60)

        # 1. Preprocesar
        print("\n⏳ 1. Preprocesando datos...")
        X = self.preprocesar(datos_usuario)
        print(f"   ✅ Datos preprocesados (shape: {X.shape})")

        # 2. Clasificar
        print("\n⏳ 2. Clasificando nivel de adicción...")
        clase_idx, clase_nombre, probs = self.clasificador.inferir(X)
        confianza = probs[clase_idx] if isinstance(probs, list) else max(probs)
        print(f"   ✅ Clase: {clase_nombre} (confianza: {confianza:.2%})")

        # 3. Predecir riesgo
        print("\n⏳ 3. Estimando riesgo futuro...")
        riesgo, categoria_riesgo = self.predictor.inferir(X)
        print(f"   ✅ Riesgo: {categoria_riesgo} ({riesgo:.2%})")

        # 4. Explicar
        print("\n⏳ 4. Identificando factores clave...")
        ranking = self.explicador.inferir(X)
        print("   ✅ Factores clave:")
        for feat, imp in ranking[:3]:
            print(f"      - {feat}: {imp:.2%}")

        # 5. Generar recomendación
        print("\n⏳ 5. Generando recomendación personalizada...")

        # Construir valores crudos
        raw_values = {}
        for col in self.artifacts['raw_columns']:
            if col in datos_usuario:
                raw_values[col] = datos_usuario[col]
            else:
                base = col.replace('_raw', '')
                if base in datos_usuario:
                    raw_values[col] = datos_usuario[base]
                else:
                    raw_values[col] = 0.0

        for key, val in datos_usuario.items():
            if key not in raw_values:
                raw_values[key] = val

        # Construir estado para el generador
        estado = {
            'clasificacion': {
                'nombre': clase_nombre,
                'indice': clase_idx,
                'probabilidades': probs
            },
            'prediccion': {
                'riesgo': riesgo,
                'categoria': categoria_riesgo
            },
            'explicacion': {
                'ranking': ranking
            },
            'datos_usuario': raw_values
        }

        clase_normalizada = clase_idx / (len(self.class_names) - 1)  # 0-1
        X_con_clase_riesgo = np.concatenate([
            X,  # 18 features originales
            [[clase_normalizada]],  # clase normalizada (0-1)
            [[riesgo]]  # riesgo (0-1)
        ], axis=1)

        print(f"   X shape para generador: {X_con_clase_riesgo.shape}")  # Debe ser (1, 20)

        resultado_generador = self.generador.inferir(estado, X=X_con_clase_riesgo)
        texto_recomendacion = resultado_generador['texto']
        urgencia = resultado_generador.get('urgencia', riesgo)
        prioridad = resultado_generador.get('prioridad', categoria_riesgo)

        print(f"   ✅ Recomendación generada (urgencia MLP: {urgencia:.2%}, prioridad: {prioridad})")

        # 6. Construir resultado final
        resultado = {
            'datos_usuario': datos_usuario,
            'clasificacion': {
                'clase': clase_nombre,
                'confianza': confianza,
                'probabilidades': probs
            },
            'prediccion': {
                'riesgo': riesgo,
                'categoria': categoria_riesgo
            },
            'explicacion': {
                'ranking': ranking
            },
            'recomendacion': {
                'texto': texto_recomendacion,
                'urgencia': urgencia,
                'prioridad': prioridad
            }
        }

        print("\n✅ Pipeline completado!")
        return resultado

    def mostrar_resultado(self, resultado):
        """Muestra el resultado en formato legible."""
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DEL ANÁLISIS")
        print("=" * 60)

        print(f"\n📋 NIVEL DE ADICCIÓN: {resultado['clasificacion']['clase'].upper()}")
        print(f"   Confianza: {resultado['clasificacion']['confianza']:.2%}")

        print(f"\n⚠️ RIESGO FUTURO: {resultado['prediccion']['categoria']}")
        print(f"   Score: {resultado['prediccion']['riesgo']:.2%}")

        if 'urgencia' in resultado['recomendacion']:
            print(f"\n🔥 URGENCIA ESTIMADA (GeneradorMLP): {resultado['recomendacion']['urgencia']:.2%}")

        print("\n🔍 FACTORES CLAVE:")
        for i, (feat, imp) in enumerate(resultado['explicacion']['ranking'][:5], 1):
            print(f"   {i}. {feat}: {imp:.2%}")

        print("\n💡 RECOMENDACIÓN PERSONALIZADA:")
        print("-" * 50)
        print(resultado['recomendacion']['texto'])
        print("-" * 50)

        print("\n" + "=" * 60)

    def guardar_resultado(self, resultado, filename=None):
        """Guarda el resultado en un archivo JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resultado_{timestamp}.json"

        resultado_serializable = {
            'datos_usuario': resultado['datos_usuario'],
            'clasificacion': resultado['clasificacion'],
            'prediccion': resultado['prediccion'],
            'explicacion': {
                'ranking': [(feat, float(imp)) for feat, imp in resultado['explicacion']['ranking']]
            },
            'recomendacion': {
                'texto': resultado['recomendacion']['texto'],
                'urgencia': float(resultado['recomendacion'].get('urgencia', 0.0)),
                'prioridad': resultado['recomendacion'].get('prioridad', 'Medio')
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(resultado_serializable, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Resultado guardado en: {filename}")
        return filename


# ============================================================================
# FUNCIONES INTERACTIVAS
# ============================================================================

def interactivo():
    print("\n" + "=" * 60)
    print("📱 MODO INTERACTIVO - INGRESA TUS DATOS")
    print("=" * 60)
    print("\nResponde las siguientes preguntas sobre tus hábitos digitales:")
    print("-" * 60)

    try:
        datos = {
            'daily_screen_time_hours': float(input("📱 Horas de pantalla al día: ")),
            'social_media_hours': float(input("📱 Horas en redes sociales al día: ")),
            'gaming_hours': float(input("🎮 Horas en juegos al día: ")),
            'work_study_hours': float(input("📚 Horas de trabajo/estudio al día: ")),
            'sleep_hours': float(input("😴 Horas de sueño al día: ")),
            'notifications_per_day': float(input("🔔 Notificaciones al día: ")),
            'app_opens_per_day': float(input("📱 Aperturas de apps al día: ")),
            'weekend_screen_time': float(input("📱 Horas de pantalla en fin de semana: "))
        }
        return datos
    except ValueError:
        print("\n❌ Error: Por favor ingresa números válidos.")
        return None


def cargar_desde_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        required = ['daily_screen_time_hours', 'sleep_hours', 'social_media_hours', 'weekend_screen_time']
        if not all(k in datos for k in required):
            print("⚠️  El archivo JSON debe contener al menos: " + ", ".join(required))
            return None
        return datos
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {filename}")
        return None
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return None


def usar_ejemplo():
    return {
        'daily_screen_time_hours': 7.5,
        'social_media_hours': 3.0,
        'gaming_hours': 1.5,
        'work_study_hours': 4.0,
        'sleep_hours': 6.0,
        'notifications_per_day': 120,
        'app_opens_per_day': 80,
        'weekend_screen_time': 9.0
    }


def main():
    sistema = RadarDigital()

    while True:
        print("\n" + "=" * 60)
        print("📡 RADAR DIGITAL - MENÚ PRINCIPAL")
        print("=" * 60)
        print("\n1. 🔍 Analizar hábitos (modo interactivo)")
        print("2. 📂 Cargar datos desde archivo JSON")
        print("3. 🧪 Usar ejemplo (perfil moderado)")
        print("4. 🚪 Salir")
        print("-" * 60)

        opcion = input("\nSelecciona una opción (1-4): ").strip()

        if opcion == "1":
            datos = interactivo()
            if datos:
                resultado = sistema.ejecutar(datos)
                sistema.mostrar_resultado(resultado)
                guardar = input("\n¿Guardar resultado? (s/n): ").strip().lower()
                if guardar == 's':
                    sistema.guardar_resultado(resultado)

        elif opcion == "2":
            filename = input("📂 Nombre del archivo JSON: ").strip()
            datos = cargar_desde_json(filename)
            if datos:
                resultado = sistema.ejecutar(datos)
                sistema.mostrar_resultado(resultado)
                guardar = input("\n¿Guardar resultado? (s/n): ").strip().lower()
                if guardar == 's':
                    sistema.guardar_resultado(resultado)

        elif opcion == "3":
            datos = usar_ejemplo()
            print("\n📊 Usando perfil de ejemplo:")
            for k, v in datos.items():
                print(f"   {k}: {v}")
            resultado = sistema.ejecutar(datos)
            sistema.mostrar_resultado(resultado)
            guardar = input("\n¿Guardar resultado? (s/n): ").strip().lower()
            if guardar == 's':
                sistema.guardar_resultado(resultado)

        elif opcion == "4":
            print("\n👋 ¡Hasta luego!")
            break

        else:
            print("❌ Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    main()

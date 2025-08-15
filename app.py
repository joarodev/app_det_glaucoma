# app.py
import sys
import os
import tensorflow as tf
from tensorflow import keras
from PySide6.QtWidgets import QApplication
from gradcam_visualizer import GradCAMVisualizer
from views.main_windows import MainWindow

def _resource_path(relative_path):
    """Resuelve rutas tanto en desarrollo como en ejecutable PyInstaller (onedir/onefile)."""
    try:
        base_dir = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

MODEL_PATH = _resource_path(os.path.join("model", "mobilenet_flV3_finetuning.h5"))
if not os.path.exists(MODEL_PATH):
    # fallback: raíz del proyecto en desarrollo
    alt = _resource_path("mobilenet_flV3_finetuning.h5")
    if os.path.exists(alt):
        MODEL_PATH = alt

def main():
    print("Cargando modelo... esto puede tardar unos segundos.")
    
    try:
        # Intentar cargar el modelo con manejo de errores
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("Modelo cargado exitosamente!")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Intentando con configuración alternativa...")
        
        try:
            # Intentar con custom_objects para manejar capas problemáticas
            model = keras.models.load_model(
                MODEL_PATH, 
                compile=False,
                custom_objects={}
            )
            print("Modelo cargado con configuración alternativa!")
        except Exception as e2:
            print(f"Error persistente al cargar el modelo: {e2}")
            print("\nEl modelo tiene problemas de compatibilidad de arquitectura.")
            print("Soluciones recomendadas:")
            print("1. Reentrenar el modelo con TensorFlow 2.16.1")
            print("2. Convertir el modelo a formato SavedModel (.pb)")
            print("3. Usar el modelo original con la versión exacta de TensorFlow con la que se entrenó")
            print("4. Verificar si hay capas personalizadas o modificaciones en el modelo")
            return
    
    # Usar exactamente el target layer que usabas: "Conv_1" u otro
    visualizer = GradCAMVisualizer(model, target_layer_name="Conv_1")

    app = QApplication(sys.argv)
    # Estilos agradables (QSS)
    app.setStyleSheet(
        """
        QWidget { background-color: #f7f9fc; color: #1f2937; }
        QTabWidget::pane { border: 1px solid #d1d5db; border-radius: 8px; }
        QTabBar::tab { background: #e5e7eb; padding: 8px 16px; margin: 2px; border-radius: 6px; }
        QTabBar::tab:selected { background: #10b981; color: white; }
        QPushButton { background-color: #2563eb; color: white; border-radius: 6px; padding: 8px 14px; }
        QPushButton:hover { background-color: #1d4ed8; }
        QPushButton:disabled { background-color: #9ca3af; }
        QListWidget { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; }
        QLabel#summary { background-color: #ecfeff; border: 1px solid #a5f3fc; border-radius: 6px; padding: 8px; }
        QTextEdit { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; font-family: Consolas, 'Courier New', monospace; font-size: 12px; }
        QLabel { font-size: 13px; }
        """
    )
    window = MainWindow(visualizer)
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

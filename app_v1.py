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
    # OPTIMIZACIÓN: Cargar modelo de forma asíncrona para mostrar GUI más rápido
    print("Iniciando aplicación de Detección de Glaucoma...")
    
    # Crear la aplicación GUI primero
    app = QApplication(sys.argv)
    
    # Estilos agregados (QSS)
    app.setStyleSheet(
        """
        /* Estilo principal - Entorno médico profesional */
        QWidget { 
            background-color: #f8fafc; 
            color: #1e293b; 
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
        }
        
        /* Tabs principales */
        QTabWidget::pane { 
            border: 2px solid #e2e8f0; 
            border-radius: 12px; 
            background-color: #ffffff;
            margin-top: 8px;
        }
        
        QTabBar::tab { 
            background: #f1f5f9; 
            color: #475569;
            padding: 12px 20px; 
            margin: 2px 4px 0px 4px; 
            border-radius: 8px 8px 0px 0px; 
            border: 1px solid #e2e8f0;
            font-weight: 500;
            min-width: 120px;
        }
        
        QTabBar::tab:selected { 
            background: #0ea5e9; 
            color: white; 
            border-bottom: 2px solid #0ea5e9;
        }
        
        QTabBar::tab:hover:!selected { 
            background: #e0f2fe; 
            color: #0369a1;
        }
        
        /* Botones principales */
        QPushButton { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #0ea5e9, stop:1 #0284c7); 
            color: white; 
            border: none;
            border-radius: 8px; 
            padding: 10px 18px; 
            font-weight: 600;
            font-size: 10pt;
            min-height: 20px;
        }
        
        QPushButton:hover { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #0284c7, stop:1 #0369a1); 
        }
        
        QPushButton:pressed { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #0369a1, stop:1 #0c4a6e); 
        }
        
        QPushButton:disabled { 
            background: #94a3b8; 
            color: #64748b;
        }
        
        /* Botones de acción secundaria */
        QPushButton#secondary { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #64748b, stop:1 #475569); 
        }
        
        QPushButton#secondary:hover { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #475569, stop:1 #334155); 
        }
        
        /* Botones de peligro */
        QPushButton#danger { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ef4444, stop:1 #dc2626); 
        }
        
        QPushButton#danger:hover { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #dc2626, stop:1 #b91c1c); 
        }
        
        /* Listas y contenedores */
        QListWidget { 
            background-color: #ffffff; 
            border: 2px solid #e2e8f0; 
            border-radius: 10px; 
            padding: 8px;
            alternate-background-color: #f8fafc;
            selection-background-color: #0ea5e9;
            selection-color: white;
        }
        
        QListWidget::item { 
            padding: 8px; 
            border-radius: 6px; 
            margin: 2px 0px;
        }
        
        QListWidget::item:selected { 
            background-color: #0ea5e9; 
            color: white;
        }
        
        QListWidget::item:hover:!selected { 
            background-color: #e0f2fe; 
            color: #0369a1;
        }
        
        /* Panel de resumen destacado */
        QLabel#Resumen { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ecfeff, stop:1 #cffafe); 
            border: 2px solid #22d3ee; 
            border-radius: 10px; 
            padding: 16px; 
            font-weight: 600;
            color: #0c4a6e;
            font-size: 10pt;
        }
        
        /* Área de texto */
        QTextEdit { 
            background-color: #ffffff; 
            border: 2px solid #e2e8f0; 
            border-radius: 10px; 
            padding: 12px;
            font-family: 'Consolas', 'Courier New', monospace; 
            font-size: 10pt;
            line-height: 1.4;
        }
        
        /* Labels generales */
        QLabel { 
            font-size: 11pt; 
            color: #1e293b;
        }
        
        /* Combos y controles */
        QComboBox { 
            background-color: #ffffff; 
            border: 2px solid #e2e8f0; 
            border-radius: 8px; 
            padding: 8px 12px;
            font-size: 10pt;
            min-height: 20px;
        }
        
        QComboBox:hover { 
            border-color: #0ea5e9; 
        }
        
        QComboBox::drop-down { 
            border: none; 
            width: 20px; 
        }
        
        QComboBox::down-arrow { 
            image: none; 
            border-left: 5px solid transparent; 
            border-right: 5px solid transparent; 
            border-top: 5px solid #64748b; 
            margin-right: 8px;
        }
        
        /* Scrollbars */
        QScrollBar:vertical { 
            background: #f1f5f9; 
            width: 12px; 
            border-radius: 6px; 
        }
        
        QScrollBar::handle:vertical { 
            background: #cbd5e1; 
            border-radius: 6px; 
            min-height: 20px; 
        }
        
        QScrollBar::handle:vertical:hover { 
            background: #94a3b8; 
        }
        
        /* Separadores */
        QSplitter::handle { 
            background-color: #e2e8f0; 
            border-radius: 2px; 
        }
        
        QSplitter::handle:horizontal { 
            width: 4px; 
        }
        
        QSplitter::handle:vertical { 
            height: 4px; 
        }
        """
    )
    
    # Crear ventana principal con placeholder para el modelo
    from PySide6.QtCore import QTimer
    
    # Crear ventana temporal mientras carga el modelo
    window = MainWindow(None)  # Sin modelo por ahora
    window.resize(1000, 700)
    window.show()
    
    # Función para cargar el modelo en segundo plano
    def load_model_and_initialize():
        try:
            print("Cargando modelo de IA...")
            model = keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Modelo cargado exitosamente!")
            
            # Crear visualizador
            visualizer = GradCAMVisualizer(model, target_layer_name="Conv_1")
            
            # Actualizar la ventana con el modelo
            window.set_model(visualizer)
            print("✅ Aplicación completamente inicializada!")
            
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            # Mostrar mensaje de error en la GUI
            window.show_error_message(f"Error al cargar el modelo: {e}")
    
    # Cargar modelo después de mostrar la GUI (mejora la experiencia del usuario)
    QTimer.singleShot(100, load_model_and_initialize)
    
    sys.exit(app.exec())

    app = QApplication(sys.argv)
    # Estilos agradables (QSS)
    app.setStyleSheet(
        """
        /* Estilo principal - Entorno médico profesional */
        QWidget { 
            background-color: #f8fafc; 
            color: #1e293b; 
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
        }
        
        /* Tabs principales */
        QTabWidget::pane { 
            border: 2px solid #e2e8f0; 
            border-radius: 12px; 
            background-color: #ffffff;
            margin-top: 8px;
        }
        
        QTabBar::tab { 
            background: #f1f5f9; 
            color: #475569;
            padding: 12px 20px; 
            margin: 2px 4px 0px 4px; 
            border-radius: 8px 8px 0px 0px; 
            border: 1px solid #e2e8f0;
            font-weight: 500;
            min-width: 120px;
        }
        
        QTabBar::tab:selected { 
            background: #0ea5e9; 
            color: white; 
            border-bottom: 2px solid #0ea5e9;
        }
        
        QTabBar::tab:hover:!selected { 
            background: #e0f2fe; 
            color: #0369a1;
        }
        
        /* Botones principales */
        QPushButton { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #0ea5e9, stop:1 #0284c7); 
            color: white; 
            border: none;
            border-radius: 8px; 
            padding: 10px 18px; 
            font-weight: 600;
            font-size: 10pt;
            min-height: 20px;
        }
        
        QPushButton:hover { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #0284c7, stop:1 #0369a1); 
        }
        
        QPushButton:pressed { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #0369a1, stop:1 #0c4a6e); 
        }
        
        QPushButton:disabled { 
            background: #94a3b8; 
            color: #64748b;
        }
        
        /* Botones de acción secundaria */
        QPushButton#secondary { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #64748b, stop:1 #475569); 
        }
        
        QPushButton#secondary:hover { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #475569, stop:1 #334155); 
        }
        
        /* Botones de peligro */
        QPushButton#danger { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ef4444, stop:1 #dc2626); 
        }
        
        QPushButton#danger:hover { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #dc2626, stop:1 #b91c1c); 
        }
        
        /* Listas y contenedores */
        QListWidget { 
            background-color: #ffffff; 
            border: 2px solid #e2e8f0; 
            border-radius: 10px; 
            padding: 8px;
            alternate-background-color: #f8fafc;
            selection-background-color: #0ea5e9;
            selection-color: white;
        }
        
        QListWidget::item { 
            padding: 8px; 
            border-radius: 6px; 
            margin: 2px 0px;
        }
        
        QListWidget::item:selected { 
            background-color: #0ea5e9; 
            color: white;
        }
        
        QListWidget::item:hover:!selected { 
            background-color: #e0f2fe; 
            color: #0369a1;
        }
        
        /* Panel de resumen destacado */
        QLabel#Resumen { 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #ecfeff, stop:1 #cffafe); 
            border: 2px solid #22d3ee; 
            border-radius: 10px; 
            padding: 16px; 
            font-weight: 600;
            color: #0c4a6e;
            font-size: 10pt;
        }
        
        /* Área de texto */
        QTextEdit { 
            background-color: #ffffff; 
            border: 2px solid #e2e8f0; 
            border-radius: 10px; 
            padding: 12px;
            font-family: 'Consolas', 'Courier New', monospace; 
            font-size: 10pt;
            line-height: 1.4;
        }
        
        /* Labels generales */
        QLabel { 
            font-size: 11pt; 
            color: #1e293b;
        }
        
        /* Combos y controles */
        QComboBox { 
            background-color: #ffffff; 
            border: 2px solid #e2e8f0; 
            border-radius: 8px; 
            padding: 8px 12px;
            font-size: 10pt;
            min-height: 20px;
        }
        
        QComboBox:hover { 
            border-color: #0ea5e9; 
        }
        
        QComboBox::drop-down { 
            border: none; 
            width: 20px; 
        }
        
        QComboBox::down-arrow { 
            image: none; 
            border-left: 5px solid transparent; 
            border-right: 5px solid transparent; 
            border-top: 5px solid #64748b; 
            margin-right: 8px;
        }
        
        /* Scrollbars */
        QScrollBar:vertical { 
            background: #f1f5f9; 
            width: 12px; 
            border-radius: 6px; 
        }
        
        QScrollBar::handle:vertical { 
            background: #cbd5e1; 
            border-radius: 6px; 
            min-height: 20px; 
        }
        
        QScrollBar::handle:vertical:hover { 
            background: #94a3b8; 
        }
        
        /* Separadores */
        QSplitter::handle { 
            background-color: #e2e8f0; 
            border-radius: 2px; 
        }
        
        QSplitter::handle:horizontal { 
            width: 4px; 
        }
        
        QSplitter::handle:vertical { 
            height: 4px; 
        }
        """
    )
    window = MainWindow(visualizer)
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

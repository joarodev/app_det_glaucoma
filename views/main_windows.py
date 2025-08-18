# views/main_window.py
import os
import sys
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
                               QLabel, QHBoxLayout, QListWidget, QListWidgetItem, QTabWidget,
                               QTextEdit, QSplitter, QComboBox)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from views.widgets import select_image, select_folder, confirm_delete
from utils.file_utils import open_folder, delete_detection_folder
from utils.history_utils import append_record, read_master
import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self, gradcam_visualizer):
        super().__init__()
        self.setWindowTitle("Detector glaucoma - App")
        self.gc = gradcam_visualizer
        
        # Estado de inicialización
        self.model_loaded = gradcam_visualizer is not None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Panel imagenes
        self.tab_single = QWidget()
        self.tabs.addTab(self.tab_single, "🔍 Detectar imagen")
        self._init_single_tab()

        # Panel carpeta
        self.tab_folder = QWidget()
        self.tabs.addTab(self.tab_folder, "📁 Detectar carpeta")
        self._init_folder_tab()

        # Panel detalle (overlay, heatmap, características)
        self.tab_detail = QWidget()
        self.tabs.addTab(self.tab_detail, "📊 Detalle de resultados")
        self._init_detail_tab()

        # Panel historial
        self.tab_hist = QWidget()
        self.tabs.addTab(self.tab_hist, "📋 Historial")
        self._init_history_tab()

        self.current_detail = None

    def _init_single_tab(self):
        layout = QVBoxLayout()
        # Título para la sección
        single_title = QLabel("🔍 Detección de imagen individual")
        single_title.setStyleSheet("font-weight: 600; color: #0c4a6e; font-size: 12pt; margin-bottom: 16px;")
        single_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(single_title)

        btn_load = QPushButton("Cargar imagen y detectar")
        btn_load.clicked.connect(self.on_load_image)
        layout.addWidget(btn_load)
        # Título para la vista previa
        preview_title = QLabel("🔍 Vista previa de la detección")
        preview_title.setStyleSheet("font-weight: 600; color: #0c4a6e; font-size: 12pt; margin: 16px 0px 8px 0px;")
        preview_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(preview_title)

        self.single_preview = QLabel("La predicción se mostrará aquí...")
        self.single_preview.setFixedSize(400, 300)
        self.single_preview.setAlignment(Qt.AlignCenter)
        self.single_preview.setStyleSheet("border: 2px solid #e2e8f0; border-radius: 8px; background-color: #f8fafc; color: #64748b;")
        layout.addWidget(self.single_preview)

        # acciones rápidas
        row = QHBoxLayout()
        self.btn_single_open = QPushButton("Mostrar en carpeta")
        self.btn_single_open.setObjectName("secondary")
        self.btn_single_open.clicked.connect(self.open_current_detail_folder)
        self.btn_single_delete = QPushButton("Borrar detección")
        self.btn_single_delete.setObjectName("danger")
        self.btn_single_delete.clicked.connect(self.delete_current_detail_folder)
        row.addWidget(self.btn_single_open); row.addWidget(self.btn_single_delete)
        layout.addLayout(row)
        self.tab_single.setLayout(layout)

    def _init_folder_tab(self):
        layout = QVBoxLayout()
        # Título para la sección
        folder_title = QLabel("📁 Procesamiento de carpeta de imágenes")
        folder_title.setStyleSheet("font-weight: 600; color: #0c4a6e; font-size: 12pt; margin-bottom: 16px;")
        folder_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(folder_title)

        btn_folder = QPushButton("Seleccionar carpeta y detectar todo")
        btn_folder.clicked.connect(self.on_select_folder)
        layout.addWidget(btn_folder)

        self.folder_summary = QLabel("")
        self.folder_summary.setObjectName("Resumen")
        layout.addWidget(self.folder_summary)

        self.folder_list = QListWidget()
        self.folder_list.itemDoubleClicked.connect(self.on_folder_item_double_clicked)
        layout.addWidget(self.folder_list)

        row = QHBoxLayout()
        self.btn_folder_open = QPushButton("Mostrar en carpeta elemento seleccionado")
        self.btn_folder_open.setObjectName("secondary")
        self.btn_folder_open.clicked.connect(self.open_selected_folder_from_list)
        self.btn_folder_delete = QPushButton("Eliminar elemento seleccionado")
        self.btn_folder_delete.setObjectName("danger")
        self.btn_folder_delete.clicked.connect(self.delete_selected_from_list)
        row.addWidget(self.btn_folder_open); row.addWidget(self.btn_folder_delete)
        layout.addLayout(row)
        self.tab_folder.setLayout(layout)

    def _init_detail_tab(self):
        layout = QVBoxLayout()
        
        # Título para la sección
        detail_title = QLabel("📊 Detalle completo de resultados")
        detail_title.setStyleSheet("font-weight: 600; color: #0c4a6e; font-size: 12pt; margin-bottom: 16px;")
        detail_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(detail_title)
        
        splitter = QSplitter()

        left = QWidget(); left_layout = QVBoxLayout(); left.setLayout(left_layout)
        right = QWidget(); right_layout = QVBoxLayout(); right.setLayout(right_layout)

        # Título para la sección de imágenes
        overlay_title = QLabel("🔍 Imagen con Grad-CAM superpuesto")
        overlay_title.setStyleSheet("font-weight: 600; color: #0c4a6e; font-size: 12pt; margin-bottom: 8px;")
        overlay_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(overlay_title)

        self.detail_overlay = QLabel("Overlay")
        self.detail_overlay.setAlignment(Qt.AlignCenter)
        self.detail_overlay.setMinimumSize(400, 300)
        self.detail_overlay.setStyleSheet("border: 2px solid #e2e8f0; border-radius: 8px; background-color: #f8fafc;")
        left_layout.addWidget(self.detail_overlay)

        # Título para heatmap
        heatmap_title = QLabel("🔥 Mapa de calor puro (Grad-CAM)")
        heatmap_title.setStyleSheet("font-weight: 600; color: #0c4a6e; font-size: 12pt; margin: 16px 0px 8px 0px;")
        heatmap_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(heatmap_title)

        self.detail_heatmap = QLabel("Heatmap")
        self.detail_heatmap.setAlignment(Qt.AlignCenter)
        self.detail_heatmap.setMinimumSize(400, 300)
        self.detail_heatmap.setStyleSheet("border: 2px solid #e2e8f0; border-radius: 8px; background-color: #f8fafc;")
        left_layout.addWidget(self.detail_heatmap)

        # Título para características
        features_title = QLabel("📊 Características detectadas")
        features_title.setStyleSheet("font-weight: 600; color: #0c4a6e; font-size: 12pt; margin-bottom: 8px;")
        features_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(features_title)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("Las características de la detección se mostrarán aquí...")
        right_layout.addWidget(self.detail_text)

        splitter.addWidget(left)
        splitter.addWidget(right)
        layout.addWidget(splitter)

        row = QHBoxLayout()
        self.btn_detail_open = QPushButton("Mostrar en carpeta")
        self.btn_detail_open.setObjectName("secondary")
        self.btn_detail_open.clicked.connect(self.open_current_detail_folder)
        self.btn_detail_delete = QPushButton("Borrar detección")
        self.btn_detail_delete.setObjectName("danger")
        self.btn_detail_delete.clicked.connect(self.delete_current_detail_folder)
        row.addWidget(self.btn_detail_open); row.addWidget(self.btn_detail_delete)
        layout.addLayout(row)

        self.tab_detail.setLayout(layout)

    def _init_history_tab(self):
        layout = QVBoxLayout()
        # Título para la sección
        history_title = QLabel("📋 Historial de detecciones")
        history_title.setStyleSheet("font-weight: 600; color: #0c4a6e; font-size: 12pt; margin-bottom: 16px;")
        history_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(history_title)

        # Controles de filtrado y ordenamiento
        controls = QHBoxLayout()
        
        # Etiqueta para ordenamiento
        sort_label = QLabel("Ordenar por:")
        sort_label.setStyleSheet("font-weight: 600; color: #475569;")
        controls.addWidget(sort_label)
        
        self.combo_sort = QComboBox()
        self.combo_sort.addItems([
            "📅 Fecha descendente", 
            "📅 Fecha ascendente", 
            "⚠️ Urgencia descendente", 
            "⚠️ Urgencia ascendente"
        ]) 
        controls.addWidget(self.combo_sort)
        
        # Espaciador
        controls.addStretch()
        
        # Etiqueta para filtro
        filter_label = QLabel("Filtrar por urgencia:")
        filter_label.setStyleSheet("font-weight: 600; color: #475569;")
        controls.addWidget(filter_label)
        
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Todo", "🔴 ALTA", "🟡 MEDIA", "🟢 BAJA"]) 
        controls.addWidget(self.combo_filter)
        
        layout.addLayout(controls)

        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.on_history_item_double_clicked)
        btn_refresh = QPushButton("Actualizar historial")
        btn_refresh.clicked.connect(self.refresh_history)
        btn_open = QPushButton("Mostrar en carpeta los seleccionados")
        btn_open.setObjectName("secondary")
        btn_open.clicked.connect(self.open_selected_folder)
        btn_delete = QPushButton("Eliminar los seleccionados")
        btn_delete.setObjectName("danger")
        btn_delete.clicked.connect(self.delete_selected_detection)

        layout.addWidget(btn_refresh)
        layout.addWidget(self.history_list)
        row = QHBoxLayout()
        row.addWidget(btn_open); row.addWidget(btn_delete)
        layout.addLayout(row)
        self.tab_hist.setLayout(layout)
        self.refresh_history()

    def on_load_image(self):
        path = select_image(self)
        if not path:
            return
        res = self.gc.process_image(path, output_root="resultados")
        append_record({
            "image": res["image"],
            "overlay_path": res["overlay_path"],
            "heatmap_puro_path": res["heatmap_puro_path"],
            "csv_path": res["csv_path"],
            "probabilidad": res["probabilidad"],
            "centro_x": res["centro"][0],
            "centro_y": res["centro"][1],
            "bbox_xmin": res["bbox"][0],
            "bbox_ymin": res["bbox"][1],
            "bbox_xmax": res["bbox"][2],
            "bbox_ymax": res["bbox"][3],
            "tamano_zona_activa": res["tamano_zona_activa"],
            "nivel_urgencia": res["nivel_urgencia"],
            "nivel_urgencia_label": res["nivel_urgencia_label"]
        })
        pix = QPixmap(res["overlay_path"])
        self.single_preview.setPixmap(pix.scaled(self.single_preview.size(), Qt.KeepAspectRatio))
        self.set_detail_from_result(res)
        self.tabs.setCurrentWidget(self.tab_detail)
        self.refresh_history()

    def on_select_folder(self):
        folder = select_folder(self)
        if not folder:
            return
        results = self.gc.process_folder(folder, output_root="resultados")
        # append each to master
        for r in results:
            append_record({
                "image": r["image"],
                "overlay_path": r["overlay_path"],
                "heatmap_puro_path": r["heatmap_puro_path"],
                "csv_path": r["csv_path"],
                "probabilidad": r["probabilidad"],
                "centro_x": r["centro"][0],
                "centro_y": r["centro"][1],
                "bbox_xmin": r["bbox"][0],
                "bbox_ymin": r["bbox"][1],
                "bbox_xmax": r["bbox"][2],
                "bbox_ymax": r["bbox"][3],
                "tamano_zona_activa": r["tamano_zona_activa"],
                "nivel_urgencia": r["nivel_urgencia"],
                "nivel_urgencia_label": r["nivel_urgencia_label"]
            })
        # ordenar por urgencia desc y llenar lista
        results_sorted = sorted(results, key=lambda x: x.get("nivel_urgencia", 0.0), reverse=True)
        self.folder_list.clear()
        total = len(results_sorted)
        alta = sum(1 for r in results_sorted if r.get("nivel_urgencia_label") == "ALTA")
        media = sum(1 for r in results_sorted if r.get("nivel_urgencia_label") == "MEDIA")
        baja = sum(1 for r in results_sorted if r.get("nivel_urgencia_label") == "BAJA")
        min_urg = min((r.get("nivel_urgencia", 0.0) for r in results_sorted), default=0.0)
        detected = sum(1 for r in results_sorted if r.get("probabilidad", 0.0) >= 0.5)
        min_prob = min((r.get("probabilidad", 0.0) for r in results_sorted), default=0.0)
        # Crear resumen visual con emojis y mejor formato
        summary_text = f"""
📊 RESUMEN DE DETECCIÓN:
• Total procesadas: {total} imágenes
• Glaucoma detectado (≥0.5): {detected} casos
• Nivel de urgencia:
  🔴 ALTA: {alta} | 🟡 MEDIA: {media} | 🟢 BAJA: {baja}
• Urgencia mínima: {min_urg:.3f}
• Probabilidad mínima: {min_prob:.3f}
        """.strip()
        self.folder_summary.setText(summary_text)
        for r in results_sorted:
            base = os.path.basename(r["image"]) if r.get("image") else "?"
            urgency_level = r['nivel_urgencia_label']
            urgency_emoji = "🔴" if urgency_level == "ALTA" else "🟡" if urgency_level == "MEDIA" else "🟢"
            prob = r.get('probabilidad', 0.0)
            prob_emoji = "✅" if prob >= 0.5 else "❌"
            
            label = f"{urgency_emoji} {base} | Prob: {prob:.3f} {prob_emoji} | Urg: {r['nivel_urgencia']:.3f} | {urgency_level}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, r)
            self.folder_list.addItem(item)
        if results_sorted:
            self.set_detail_from_result(results_sorted[0])
        self.refresh_history()

    def refresh_history(self):
        self.history_list.clear()
        try:
            df = read_master()
            # filtro por label si aplica
            label_filter = self.combo_filter.currentText() if hasattr(self, "combo_filter") else "Todo"
            if label_filter and label_filter != "Todo" and "nivel_urgencia_label" in df.columns:
                # Extraer solo el texto sin emojis para el filtro
                clean_filter = label_filter.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")
                df = df[df["nivel_urgencia_label"] == clean_filter]
            # ordenar según selección
            sort_mode = self.combo_sort.currentText() if hasattr(self, "combo_sort") else "📅 Fecha descendente"
            if sort_mode == "📅 Fecha ascendente" and "timestamp" in df.columns:
                df_sorted = df.sort_values(by="timestamp", ascending=True)
            elif sort_mode == "📅 Fecha descendente" and "timestamp" in df.columns:
                df_sorted = df.sort_values(by="timestamp", ascending=False)
            elif sort_mode == "⚠️ Urgencia ascendente":
                df_sorted = df.sort_values(by="nivel_urgencia", ascending=True)
            else:
                df_sorted = df.sort_values(by="nivel_urgencia", ascending=False)
            
            for _, row in df_sorted.iterrows():
                base = os.path.basename(row['image']) if 'image' in row else '?'  # corregir bug: no existe nombre_imagen en master
                urgency_level = row.get('nivel_urgencia_label', 'N/A')
                urgency_emoji = "🔴" if urgency_level == "ALTA" else "🟡" if urgency_level == "MEDIA" else "🟢" if urgency_level == "BAJA" else "⚪"
                prob = row.get('probabilidad', 0.0)
                prob_emoji = "✅" if prob >= 0.5 else "❌"
                timestamp = str(row.get('timestamp', ''))[:19] if row.get('timestamp') else 'N/A'
                
                label = f"📅 {timestamp} | {urgency_emoji} {base} | Prob: {prob:.3f} {prob_emoji} | Urg: {row.get('nivel_urgencia', 0.0):.3f} | {urgency_level}"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, row.to_dict())
                self.history_list.addItem(item)
        except Exception as e:
            print("No history or error:", e)

    def set_detail_from_result(self, res: dict):
        self.current_detail = res
        # imágenes
        try:
            if res.get("overlay_path") and os.path.exists(res["overlay_path"]):
                pix1 = QPixmap(res["overlay_path"]).scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.detail_overlay.setPixmap(pix1)
            else:
                self.detail_overlay.setText("Overlay not found")
            if res.get("heatmap_puro_path") and os.path.exists(res["heatmap_puro_path"]):
                pix2 = QPixmap(res["heatmap_puro_path"]).scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.detail_heatmap.setPixmap(pix2)
            else:
                self.detail_heatmap.setText("Heatmap not found")
        except Exception:
            pass
        # texto características
        try:
            # Crear un formato más legible para el usuario médico
            features_text = f"""📊 RESULTADO DE DETECCIÓN

🔍 IMAGEN:
   Archivo: {os.path.basename(res.get('image', 'N/A'))}

📈 PROBABILIDADES:
   • Probabilidad de glaucoma: {res.get('probabilidad', 0.0):.1%}
   • Nivel de urgencia: {res.get('nivel_urgencia_label', 'N/A')} ({res.get('nivel_urgencia', 0.0):.3f})

📍 LOCALIZACIÓN:
   • Centro de la zona activa: ({res.get('centro', (0, 0))[0]:.1f}, {res.get('centro', (0, 0))[1]:.1f})
   • Bounding box: {res.get('bbox', (0, 0, 0, 0))}

📏 CARACTERÍSTICAS:
   • Tamaño de zona activa: {res.get('tamano_zona_activa', 0.0):.1%}
   • Archivos generados:
     - Overlay: {os.path.basename(res.get('overlay_path', 'N/A'))}
     - Heatmap: {os.path.basename(res.get('heatmap_puro_path', 'N/A'))}
     - Datos CSV: {os.path.basename(res.get('csv_path', 'N/A'))}

📋 DATOS TÉCNICOS COMPLETOS:
{json.dumps({
    k: (float(v) if isinstance(v, (int, float)) else v) for k, v in res.items()
}, indent=2, ensure_ascii=False)}
            """.strip()
            self.detail_text.setPlainText(features_text)
        except Exception:
            self.detail_text.setPlainText(str(res))

    def on_folder_item_double_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if isinstance(data, dict):
            self.set_detail_from_result(data)
            self.tabs.setCurrentWidget(self.tab_detail)

    def on_history_item_double_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if isinstance(data, dict):
            # map row dict to expected keys
            # prefer existing overlay_path/heatmap paths
            mapped = {
                "image": data.get("image"),
                "overlay_path": data.get("overlay_path"),
                "heatmap_puro_path": data.get("heatmap_puro_path"),
                "csv_path": data.get("csv_path"),
                "probabilidad": data.get("probabilidad", 0.0),
                "centro": (data.get("centro_x", -1), data.get("centro_y", -1)),
                "bbox": (
                    data.get("bbox_xmin", -1),
                    data.get("bbox_ymin", -1),
                    data.get("bbox_xmax", -1),
                    data.get("bbox_ymax", -1),
                ),
                "tamano_zona_activa": data.get("tamano_zona_activa", 0.0),
                "nivel_urgencia": data.get("nivel_urgencia", 0.0),
                "nivel_urgencia_label": data.get("nivel_urgencia_label", "-")
            }
            self.set_detail_from_result(mapped)
            self.tabs.setCurrentWidget(self.tab_detail)

    def open_selected_folder(self):
        item = self.history_list.currentItem()
        if not item:
            return
        info = item.data(Qt.UserRole)
        folder = os.path.join("resultados", os.path.splitext(os.path.basename(info["image"]))[0])
        if os.path.exists(folder):
            open_folder(folder)

    def delete_selected_detection(self):
        item = self.history_list.currentItem()
        if not item:
            return
        info = item.data(Qt.UserRole)
        folder = os.path.join("resultados", os.path.splitext(os.path.basename(info["image"]))[0])
        if confirm_delete(self, folder):
            deleted = delete_detection_folder(folder)
            if deleted:
                self.refresh_history()

    def open_current_detail_folder(self):
        if not self.current_detail:
            return
        img_path = self.current_detail.get("image")
        if not img_path:
            return
        folder = os.path.join("resultados", os.path.splitext(os.path.basename(img_path))[0])
        if os.path.exists(folder):
            open_folder(folder)

    def delete_current_detail_folder(self):
        if not self.current_detail:
            return
        img_path = self.current_detail.get("image")
        if not img_path:
            return
        folder = os.path.join("resultados", os.path.splitext(os.path.basename(img_path))[0])
        if confirm_delete(self, folder):
            if delete_detection_folder(folder):
                self.refresh_history()

    def open_selected_folder_from_list(self):
        item = self.folder_list.currentItem()
        if not item:
            return
        res = item.data(Qt.UserRole)
        if not isinstance(res, dict):
            return
        img_path = res.get("image")
        if not img_path:
            return
        folder = os.path.join("resultados", os.path.splitext(os.path.basename(img_path))[0])
        if os.path.exists(folder):
            open_folder(folder)

    def delete_selected_from_list(self):
        item = self.folder_list.currentItem()
        if not item:
            return
        res = item.data(Qt.UserRole)
        if not isinstance(res, dict):
            return
        img_path = res.get("image")
        if not img_path:
            return
        folder = os.path.join("resultados", os.path.splitext(os.path.basename(img_path))[0])
        if confirm_delete(self, folder):
            if delete_detection_folder(folder):
                self.folder_list.takeItem(self.folder_list.currentRow())
    
    def set_model(self, gradcam_visualizer):
        """Actualiza el modelo después de la carga inicial"""
        self.gc = gradcam_visualizer
        self.model_loaded = True
        print("✅ Modelo configurado en la ventana principal")
    
    def show_error_message(self, message):
        """Muestra un mensaje de error en la interfaz"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", message)

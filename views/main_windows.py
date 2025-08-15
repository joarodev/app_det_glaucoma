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

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Panel imagenes
        self.tab_single = QWidget()
        self.tabs.addTab(self.tab_single, "Detectar imagen")
        self._init_single_tab()

        # Panel carpeta
        self.tab_folder = QWidget()
        self.tabs.addTab(self.tab_folder, "Detectar carpeta de imagenes")
        self._init_folder_tab()

        # Panel detalle (overlay, heatmap, características)
        self.tab_detail = QWidget()
        self.tabs.addTab(self.tab_detail, "Detalle de resultados")
        self._init_detail_tab()

        # Panel historial
        self.tab_hist = QWidget()
        self.tabs.addTab(self.tab_hist, "Historial")
        self._init_history_tab()

        self.current_detail = None

    def _init_single_tab(self):
        layout = QVBoxLayout()
        btn_load = QPushButton("Cargar imagen y detectar")
        btn_load.clicked.connect(self.on_load_image)
        layout.addWidget(btn_load)
        self.single_preview = QLabel("La predicción se mostrará aquí...")
        self.single_preview.setFixedSize(400, 300)
        self.single_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.single_preview)

        # acciones rápidas
        row = QHBoxLayout()
        self.btn_single_open = QPushButton("Mostrar en carpeta")
        self.btn_single_open.clicked.connect(self.open_current_detail_folder)
        self.btn_single_delete = QPushButton("Borrar detección")
        self.btn_single_delete.clicked.connect(self.delete_current_detail_folder)
        row.addWidget(self.btn_single_open); row.addWidget(self.btn_single_delete)
        layout.addLayout(row)
        self.tab_single.setLayout(layout)

    def _init_folder_tab(self):
        layout = QVBoxLayout()
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
        self.btn_folder_open.clicked.connect(self.open_selected_folder_from_list)
        self.btn_folder_delete = QPushButton("Eliminar elemento seleccionado")
        self.btn_folder_delete.clicked.connect(self.delete_selected_from_list)
        row.addWidget(self.btn_folder_open); row.addWidget(self.btn_folder_delete)
        layout.addLayout(row)
        self.tab_folder.setLayout(layout)

    def _init_detail_tab(self):
        layout = QVBoxLayout()
        splitter = QSplitter()

        left = QWidget(); left_layout = QVBoxLayout(); left.setLayout(left_layout)
        right = QWidget(); right_layout = QVBoxLayout(); right.setLayout(right_layout)

        self.detail_overlay = QLabel("Overlay")
        self.detail_overlay.setAlignment(Qt.AlignCenter)
        self.detail_overlay.setMinimumSize(400, 300)
        self.detail_heatmap = QLabel("Heatmap")
        self.detail_heatmap.setAlignment(Qt.AlignCenter)
        self.detail_heatmap.setMinimumSize(400, 300)
        left_layout.addWidget(self.detail_overlay)
        left_layout.addWidget(self.detail_heatmap)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        right_layout.addWidget(self.detail_text)

        splitter.addWidget(left)
        splitter.addWidget(right)
        layout.addWidget(splitter)

        row = QHBoxLayout()
        self.btn_detail_open = QPushButton("Mostrar en carpeta")
        self.btn_detail_open.clicked.connect(self.open_current_detail_folder)
        self.btn_detail_delete = QPushButton("Borrar detección")
        self.btn_detail_delete.clicked.connect(self.delete_current_detail_folder)
        row.addWidget(self.btn_detail_open); row.addWidget(self.btn_detail_delete)
        layout.addLayout(row)

        self.tab_detail.setLayout(layout)

    def _init_history_tab(self):
        layout = QVBoxLayout()
        controls = QHBoxLayout()
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Filtrar: fecha descendente", "Sort: fecha ascendente", "Sort: urgencia descendente", "Sort: urgencia ascendente"]) 
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Todo", "ALTA", "MEDIA", "BAJA"]) 
        controls.addWidget(self.combo_sort); controls.addWidget(self.combo_filter)
        layout.addLayout(controls)

        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.on_history_item_double_clicked)
        btn_refresh = QPushButton("Actualizar historial")
        btn_refresh.clicked.connect(self.refresh_history)
        btn_open = QPushButton("Mostrar en carpeta los seleccionados")
        btn_open.clicked.connect(self.open_selected_folder)
        btn_delete = QPushButton("Eliminar los seleccionados")
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
        self.folder_summary.setText(
            f"Processed: {total} | Detected>=0.5: {detected} | ALTA: {alta} | MEDIA: {media} | BAJA: {baja} | min urgency: {min_urg:.3f} | min prob: {min_prob:.3f}"
        )
        for r in results_sorted:
            base = os.path.basename(r["image"]) if r.get("image") else "?"
            label = f"{base} | urg={r['nivel_urgencia']:.3f} | {r['nivel_urgencia_label']}"
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
            label_filter = self.combo_filter.currentText() if hasattr(self, "combo_filter") else "All"
            if label_filter and label_filter != "All" and "nivel_urgencia_label" in df.columns:
                df = df[df["nivel_urgencia_label"] == label_filter]
            # ordenar según selección
            sort_mode = self.combo_sort.currentText() if hasattr(self, "combo_sort") else "Sort: urgency desc"
            if sort_mode == "Sort: date asc" and "timestamp" in df.columns:
                df_sorted = df.sort_values(by="timestamp", ascending=True)
            elif sort_mode == "Sort: date desc" and "timestamp" in df.columns:
                df_sorted = df.sort_values(by="timestamp", ascending=False)
            elif sort_mode == "Sort: urgency asc":
                df_sorted = df.sort_values(by="nivel_urgencia", ascending=True)
            else:
                df_sorted = df.sort_values(by="nivel_urgencia", ascending=False)
            for _, row in df_sorted.iterrows():
                base = os.path.basename(row['image']) if 'image' in row else '?'  # corregir bug: no existe nombre_imagen en master
                label = f"{str(row['timestamp'])[:19]} | {base} | urg={row['nivel_urgencia']:.3f} | {row['nivel_urgencia_label']}"
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
            pretty = json.dumps({
                k: (float(v) if isinstance(v, (int, float)) else v) for k, v in res.items()
            }, indent=2, ensure_ascii=False)
            self.detail_text.setPlainText(pretty)
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

# views/widgets.py
from PySide6.QtWidgets import QFileDialog, QMessageBox
import os

def select_image(parent=None):
    fname, _ = QFileDialog.getOpenFileName(parent, "Seleccionar imagen", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
    return fname

def select_folder(parent=None):
    folder = QFileDialog.getExistingDirectory(parent, "Seleccionar carpeta")
    return folder

def confirm_delete(parent, folder):
    reply = QMessageBox.question(parent, "Eliminar detección", f"¿Eliminar la carpeta {os.path.basename(folder)}?", QMessageBox.Yes | QMessageBox.No)
    return reply == QMessageBox.Yes

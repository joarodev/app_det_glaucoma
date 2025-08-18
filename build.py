#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build limpio para empaquetar la app de Detección de Glaucoma con PyInstaller
(sin exclusiones y con consola para debug)
"""

import os
import sys
import subprocess

def build_executable():
    print("🚀 Iniciando build limpio con consola...")

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',                     # Un solo archivo
        '--name=Deteccion_Glaucoma',     # Nombre ejecutable
        '--icon=icon.ico',               # Icono
        '--add-data=model;model',        # Incluir modelo
        '--add-data=utils;utils',        # Incluir utilidades
        '--add-data=views;views',        # Incluir vistas

        # Forzamos hidden-imports para TF + Keras
        '--hidden-import=tensorflow',
        '--hidden-import=tensorflow._api.v2',
        '--hidden-import=tensorflow.python',
        '--hidden-import=tensorflow.keras',
        '--hidden-import=tensorflow.keras.models',
        '--hidden-import=tensorflow.keras.preprocessing',
        '--hidden-import=tensorflow.keras.utils',
        '--hidden-import=tensorflow.keras.layers',
        '--hidden-import=tensorflow.keras.backend',
        '--hidden-import=PIL',
        '--hidden-import=cv2',
        '--hidden-import=PySide6',
        '--hidden-import=gradcam_visualizer'
    ]

    # Archivo principal
    cmd.append('app_v2.py')

    try:
        print("Comando:", " ".join(cmd))
        result = subprocess.run(cmd, check=True)

        if result.returncode == 0:
            print("\n✅ Build completado con éxito")
            print("📁 Ejecutable en: dist/Deteccion_Glaucoma.exe")
            print("⚠️ Abre el exe desde consola para ver posibles errores en runtime")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en PyInstaller: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def main():
    print("=" * 60)
    print(" BUILD LIMPIO (DEBUG) ")
    print("=" * 60)
    build_executable()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de empaquetado con ICONO personalizado y optimizaciones
"""

import os
import sys
import subprocess

def main():
    print("🚀 Empaquetado con ICONO personalizado y optimizaciones")
    print("=" * 60)
    
    # Verificar que existe el icono
    if not os.path.exists("icon.ico"):
        print("❌ No se encontró icon.ico")
        print("Ejecuta primero: python create_icon.py")
        return
    
    print("✅ Icono encontrado: icon.ico")
    
    # Comando de PyInstaller con icono y optimizaciones
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',                    # Un solo archivo
        '--windowed',                   # Sin consola
        '--name=Deteccion_Glaucoma',    # Nombre del ejecutable
        '--icon=icon.ico',              # ICONO PERSONALIZADO
        '--add-data=model;model',       # Incluir modelo
        '--add-data=utils;utils',       # Incluir utilidades
        '--add-data=views;views',       # Incluir vistas
        
        # Módulos ocultos esenciales
        '--hidden-import=tensorflow',
        '--hidden-import=tensorflow.keras',
        '--hidden-import=PIL',
        '--hidden-import=cv2',
        '--hidden-import=PySide6',
        '--hidden-import=gradcam_visualizer',
        '--hidden-import=inspect',
        '--hidden-import=traitlets',
        
        # OPTIMIZACIONES PARA VELOCIDAD
        '--exclude-module=jupyter',
        '--exclude-module=notebook',
        '--exclude-module=ipython',
        '--exclude-module=matplotlib.tests',
        '--exclude-module=numpy.tests',
        '--exclude-module=pandas.tests',
        '--exclude-module=sklearn.tests',
        '--exclude-module=tensorflow.tests',
        '--exclude-module=email',
        '--exclude-module=http',
        '--exclude-module=urllib',
        '--exclude-module=xml',
        '--exclude-module=pydoc',
        '--exclude-module=doctest',
        '--exclude-module=argparse',
        '--exclude-module=tkinter',
        '--exclude-module=test',
        '--exclude-module=unittest',
        '--exclude-module=pdb',
        '--exclude-module=profile',
        '--exclude-module=pstats',
        '--exclude-module=timeit',
        '--exclude-module=trace',
        '--exclude-module=turtle',
        '--exclude-module=webbrowser',
        '--exclude-module=zipfile',
        '--exclude-module=py',
        '--exclude-module=pytest',
        '--exclude-module=setuptools',
        '--exclude-module=distutils',
        '--exclude-module=pip',
        '--exclude-module=wheel',
        
        # Configuraciones de velocidad
        '--optimize=2',                 # Optimización máxima de Python
        
        'app_clean.py'                  # Usar archivo limpio
    ]
    
    try:
        print("Iniciando empaquetado con icono...")
        print("Comando:", " ".join(cmd))
        
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print("\n✅ ¡Empaquetado con ICONO completado!")
            print(f"📁 Ejecutable: dist/Deteccion_Glaucoma.exe")
            print("\n🎨 CARACTERÍSTICAS:")
            print("  - Icono personalizado incluido")
            print("  - Módulos innecesarios excluidos")
            print("  - Optimización Python nivel 2")
            print("  - Carga asíncrona del modelo")
            print("  - Solo módulos esenciales incluidos")
            
            # Verificar tamaño del ejecutable
            exe_path = 'dist/Deteccion_Glaucoma.exe'
            if os.path.exists(exe_path):
                size_mb = os.path.getsize(exe_path) / (1024 * 1024)
                print(f"\n📊 Tamaño del ejecutable: {size_mb:.1f} MB")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()

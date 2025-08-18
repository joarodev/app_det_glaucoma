# 🧩 APP_DET_GLAUCOMA

Aplicación de escritorio para la **detección asistida de glaucoma**
mediante redes neuronales profundas.\
Desarrollada en **Python, PySide6 y TensorFlow/Keras**, la aplicación
permite cargar imágenes clínicas y obtener predicciones acompañadas de
visualizaciones interpretables con **Grad-CAM**.

------------------------------------------------------------------------

## 📋 Requisitos previos

-   Python **3.11.13**
-   Se recomienda el uso de **entorno virtual** (virtualenv o venv).
-   Instalar dependencias desde `requirements.txt`:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Ejecutar la aplicación (modo desarrollo)

1.  Clonar el repositorio o descargar el proyecto.
2.  Activar el entorno virtual.
3.  Ejecutar:

``` bash
python app_v2.py
```

------------------------------------------------------------------------

## 📦 Generar el ejecutable (.exe)

El proyecto incluye scripts para construir la aplicación en formato
ejecutable usando **PyInstaller**.

1.  Limpiar compilaciones previas:

``` bash
python build_clean.py
```

2.  Construir con icono personalizado:

``` bash
python build_with_icon.py
```

3.  El ejecutable se encontrará en la carpeta:

```{=html}
/dist/Deteccion_Glaucoma.exe
```

------------------------------------------------------------------------

## 🧠 Resumen del modelo de glaucoma

El modelo de detección de glaucoma fue entrenado utilizando
**MobileNetV2** como red base, con técnicas de *fine-tuning* y *data
augmentation* agresivo para mejorar la generalización.\
Se optimizó con el optimizador **Adam**, aplicando balanceo de clases
(`class_weight`) y estrategias de reducción de *learning rate*.


El modelo no se encuentra incluido en el repositorio por razones de
tamaño y licencia.\
En caso de requerir acceso, por favor ponerse en contacto conmigo vía
**LinkedIn o email** indicando el uso que se le dará a la aplicación.

------------------------------------------------------------------------

## 🔬 Cómo se construyó el modelo

1.  **Búsqueda de datasets clínicos** de imágenes oculares con y sin
    glaucoma.\
2.  **Preprocesamiento**: limpieza, normalización y aumentación de
    datos.\
3.  **Entrenamiento con MobileNetV2**, congelando capas base y luego
    aplicando *fine-tuning*.\
4.  **Validación cruzada** para evitar *overfitting*.\
5.  **Grad-CAM** implementado para interpretar la activación de la red
    en cada predicción.

Este flujo asegura que el sistema no solo clasifique, sino que también
brinde **explicaciones visuales** para su interpretación clínica.

------------------------------------------------------------------------

## 📂 Estructura del proyecto

    APP_DET_GLAUCOMA/
    │── build/                   # Archivos de compilación temporal
    │── dist/                    # Ejecutables generados
    │   └── Deteccion_Glaucoma.exe
    │── model/                   # Modelos entrenados (no incluidos en GitHub)
    │── utils/                   # Utilidades y scripts de apoyo
    │── views/                   # Interfaces gráficas (PySide6)
    │── app_v1.py                # Versión inicial de la app
    │── app_v2.py                # Versión principal (recomendada)
    │── gradcam_visualizer.py    # Clase para interpretar predicciones con Grad-CAM
    │── build.py                 # Script para construir la app .exe (recomendable)
    │── build_with_icon.py       # Script para construir .exe con icono
    │── requirements.txt         # Dependencias del proyecto
    │── README.md                # Documentación

------------------------------------------------------------------------

## 📬 Contacto

Para consultas o solicitud del modelo entrenado, puedes contactarme en:

-   📧 **Email:** \[joaquinrodriguez.dev@gmail.com\]
-   🔗 **LinkedIn:** \[https://www.linkedin.com/in/joaquinrodriguez-dev/\]

------------------------------------------------------------------------

📌 *Este proyecto fue desarrollado como apoyo en la investigación de
detección de glaucoma con IA, y no reemplaza la evaluación de un
profesional de la salud.*
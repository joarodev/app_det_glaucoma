#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para crear un icono personalizado para la aplicación de glaucoma
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_glaucoma_icon():
    """Crea un icono personalizado para la aplicación"""
    
    # Crear imagen base (256x256 para alta resolución)
    size = 256
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Colores médicos profesionales
    primary_color = (14, 165, 233)    # Azul médico
    secondary_color = (34, 211, 238)  # Azul claro
    accent_color = (236, 254, 255)    # Azul muy claro
    
    # Dibujar círculo principal (ojo)
    center = size // 2
    eye_radius = size // 3
    draw.ellipse(
        [center - eye_radius, center - eye_radius, 
         center + eye_radius, center + eye_radius],
        fill=accent_color,
        outline=primary_color,
        width=8
    )
    
    # Dibujar pupila
    pupil_radius = eye_radius // 2
    draw.ellipse(
        [center - pupil_radius, center - pupil_radius,
         center + pupil_radius, center + pupil_radius],
        fill=primary_color
    )
    
    # Dibujar reflejo en el ojo
    highlight_radius = pupil_radius // 3
    highlight_x = center - eye_radius // 3
    highlight_y = center - eye_radius // 3
    draw.ellipse(
        [highlight_x - highlight_radius, highlight_y - highlight_radius,
         highlight_x + highlight_radius, highlight_y + highlight_radius],
        fill=(255, 255, 255, 180)
    )
    
    # Agregar texto "G" (Glaucoma)
    try:
        # Intentar usar una fuente del sistema
        font_size = size // 4
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback a fuente por defecto
        font = ImageFont.load_default()
    
    # Dibujar "G" en la esquina inferior derecha
    text = "G"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = center + eye_radius // 2
    text_y = center + eye_radius // 2
    
    # Fondo para el texto
    padding = 10
    draw.rounded_rectangle(
        [text_x - padding, text_y - padding,
         text_x + text_width + padding, text_y + text_height + padding],
        radius=15,
        fill=primary_color
    )
    
    # Dibujar texto
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    # Guardar en diferentes tamaños
    icon_sizes = [16, 32, 48, 64, 128, 256]
    
    for icon_size in icon_sizes:
        resized_img = img.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        resized_img.save(f"icon_{icon_size}.png", "PNG")
    
    # Guardar como ICO (formato de icono de Windows)
    img.save("icon.ico", format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    
    print("✅ Iconos creados exitosamente!")
    print("📁 Archivos generados:")
    for icon_size in icon_sizes:
        print(f"   - icon_{icon_size}.png ({icon_size}x{icon_size})")
    print(f"   - icon.ico (formato Windows)")
    print("\n💡 Ahora puedes usar 'icon.ico' en tu empaquetado")

if __name__ == "__main__":
    create_glaucoma_icon()

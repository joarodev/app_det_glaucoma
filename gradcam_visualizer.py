import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class GradCAMVisualizer:
    # Constructor de la clase GradCAMVisualizer
    # Recibe un modelo secuencial y el nombre de la capa objetivo (opcional
    def __init__(self, sequential_model, target_layer_name=None):
        self.sequential_model = sequential_model
        self.base_model = sequential_model.layers[0]  # Modelo funcional MobileNetV2

        # Forzar ejecución para que input/output se definan
        dummy = tf.zeros((1, 224, 224, 3))
        _ = sequential_model(dummy)

        # Buscar capa convolucional objetivo en el base_model
        if target_layer_name is None:
            conv_layers = [layer for layer in self.base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
            if not conv_layers:
                raise ValueError("No se encontró capa convolucional en base_model")
            self.target_layer = conv_layers[-1]
        else:
            self.target_layer = self.base_model.get_layer(target_layer_name)
        print(f"[INFO] Usando capa objetivo: {self.target_layer.name}")

        # Crear modelo para Grad-CAM:
        # input: base_model.input
        # output: [target_layer.output, output_final]
        # Pero la salida final la obtendremos pasando la salida del base_model
        # por las capas restantes del modelo secuencial
        base_output = self.base_model.output
        x = base_output
        # Las capas fuera del base_model comienzan desde la posición 1 del secuencial
        for layer in sequential_model.layers[1:]:
            x = layer(x)
        output_final = x

        self.grad_model = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=[self.target_layer.output, output_final]
        )
    # ------------------------------------------------------------------

    def compute_heatmap(self, img_input, class_index=None):
        """
        img_input: tensor numpy shape (1, H, W, 3), normalizado 0-1 (float32)
        devuelve: heatmap (2D numpy float, normalized 0..1), probabilidad (float)
        """
        # grad_model ya fue creado en el constructor y usa base_model.input y salida final
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(tf.convert_to_tensor(img_input, dtype=tf.float32))
            # Elegir clase: si salida es escalar, usamos índice 0; si multi-clase usamos argmax
            if class_index is None:
                if predictions.shape[-1] == 1:
                    class_index = 0
                else:
                    class_index = tf.argmax(predictions[0]).numpy()
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)  # gradientes w.r.t. activations
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # promedio espacial por canal

        conv_outputs = conv_outputs[0]  # quitar batch dim -> (h, w, channels)
        # Grad-CAM: ponderar cada mapa de activación por su gradiente promedio y sumar
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

        # normalizar y asegurar no-negativos
        heatmap = np.maximum(heatmap.numpy(), 0)
        maxv = heatmap.max() if heatmap.max() != 0 else 1e-10
        heatmap = heatmap / maxv

        # obtener probabilidad de la clase seleccionada (si salida única, predictions[0][0])
        if predictions.shape[-1] == 1:
            prob = float(predictions.numpy()[0][0])
        else:
            prob = float(predictions.numpy()[0][class_index])

        return heatmap, prob

    def _resize_heatmap(self, heatmap, target_shape):
        """
        redimensiona heatmap (2D) a tamaño target_shape (h, w) usando INTER_CUBIC
        """
        h, w = target_shape
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        # garantizar rango 0..1
        heatmap_resized = np.clip(heatmap_resized, 0, 1)
        return heatmap_resized

    def compute_active_zone(self, heatmap_resized, threshold=0.7):
        """
        heatmap_resized: heatmap ya redimensionado al tamaño de la imagen original (h,w), rango 0..1
        threshold: umbral para considerar zona activa
        devuelve: area_ratio (0..1), center_xy (x, y) en pixeles, bbox (xmin,ymin,xmax,ymax) en pixeles, mask (bool array)
        """
        mask = heatmap_resized >= threshold
        total_pixels = mask.size
        active_pixels = int(np.sum(mask))
        area_ratio = active_pixels / total_pixels if total_pixels > 0 else 0.0

        if active_pixels == 0:
            # sin zona activa
            center_x = -1
            center_y = -1
            bbox = (-1, -1, -1, -1)
        else:
            coords = np.argwhere(mask)  # [[y,x],...]
            ys = coords[:, 0]
            xs = coords[:, 1]
            center_y = float(np.mean(ys))
            center_x = float(np.mean(xs))
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            bbox = (xmin, ymin, xmax, ymax)

        return area_ratio, (center_x, center_y), bbox, mask

    def create_overlay(self, original_img_rgb, heatmap_resized, circle_center=None, circle_radius=20, bbox=None, alpha=0.4):
        """
        original_img_rgb: imagen original en RGB (H,W,3) uint8 o float 0..255
        heatmap_resized: heatmap 2D 0..1 en tamaño HxW
        circle_center: (x,y) para dibujar círculo
        bbox: (xmin,ymin,xmax,ymax) para dibujar rectángulo
        alpha: peso del heatmap
        devuelve: overlay_rgb (uint8), heatmap_color (BGR uint8)
        """
        # convertir heatmap a color
        hm = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)  # BGR
        # original_img_rgb -> convertir a BGR para cv2.addWeighted, luego volver a RGB
        orig_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(hm_color, alpha, orig_bgr, 1 - alpha, 0)

        # dibujar marcadores sobre overlay (en BGR)
        if circle_center is not None and circle_center[0] >= 0:
            cx, cy = int(round(circle_center[0])), int(round(circle_center[1]))
            cv2.circle(overlay_bgr, (cx, cy), circle_radius, (0, 255, 255), 3)  # amarillo BGR
        if bbox is not None and bbox[0] >= 0:
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(overlay_bgr, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)  # blanco

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        return overlay_rgb, hm_color  # hm_color es BGR

    def process_image(self, image_path, output_root="resultados", threshold=0.7, circle_radius=25, save_images=True):
        """
        Procesa 1 imagen y guarda resultados en output_root/<nombre_sin_ext>/
        Guarda: overlay (predicción con gradcam), gradcam_puro (colormap), archivo CSV con datos.
        Retorna un diccionario con los valores clave.
        """
        # leer imagen original (BGR) y convertir a RGB
        orig_bgr = cv2.imread(image_path)
        if orig_bgr is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_rgb.shape[:2]

        # preprocesar para el modelo (224x224 y normalizar)
        img_resized_for_model = cv2.resize(orig_rgb, (224, 224))
        img_input = np.expand_dims(img_resized_for_model.astype(np.float32) / 255.0, axis=0)

        # calcular heatmap y prob
        heatmap_small, prob = self.compute_heatmap(img_input)

        # redimensionar heatmap a tamaño original
        heatmap_resized = self._resize_heatmap(heatmap_small, (orig_h, orig_w))

        # calcular zona activa (sobre el heatmap redimensionado)
        area_ratio, (center_x, center_y), bbox, mask = self.compute_active_zone(heatmap_resized, threshold=threshold)

        # calcular nivel de urgencia (promedio entre prob y area_ratio)
        urgency = float((prob + area_ratio) / 2.0)
        # etiqueta rápida de urgencia
        if urgency >= 0.75:
            urgency_label = "ALTA"
        elif urgency >= 0.5:
            urgency_label = "MEDIA"
        else:
            urgency_label = "BAJA"

        # crear carpeta resultado
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_folder = os.path.join(output_root, base_name)
        os.makedirs(out_folder, exist_ok=True)

        # crear_overlay y guardar imágenes
        overlay_rgb, heatmap_color_bgr = self.create_overlay(orig_rgb, heatmap_resized, circle_center=(center_x, center_y), circle_radius=circle_radius, bbox=bbox, alpha=0.45)

        if save_images:
            # overlay (RGB -> BGR for saving)
            overlay_bgr = cv2.cvtColor(overlay_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            overlay_path = os.path.join(out_folder, f"{base_name}_overlay.png")
            cv2.imwrite(overlay_path, overlay_bgr)

            # heatmap puro: heatmap_color_bgr (BGR)
            heatmap_puro_path = os.path.join(out_folder, f"{base_name}_heatmap_puro.png")
            cv2.imwrite(heatmap_puro_path, heatmap_color_bgr)

        # Guardar CSV con datos
        csv_path = os.path.join(out_folder, f"{base_name}_datos.csv")
        df = pd.DataFrame([{
            "nombre_imagen": os.path.basename(image_path),
            "probabilidad": float(prob),
            "centro_x": float(center_x),
            "centro_y": float(center_y),
            "bbox_xmin": int(bbox[0]) if bbox[0] is not None else -1,
            "bbox_ymin": int(bbox[1]) if bbox[1] is not None else -1,
            "bbox_xmax": int(bbox[2]) if bbox[2] is not None else -1,
            "bbox_ymax": int(bbox[3]) if bbox[3] is not None else -1,
            "tamano_zona_activa": float(area_ratio),
            "nivel_urgencia": float(urgency),
            "nivel_urgencia_label": urgency_label
        }])
        df.to_csv(csv_path, index=False)

        # retornar resumen
        return {
            "image": image_path,
            "overlay_path": overlay_path if save_images else None,
            "heatmap_puro_path": heatmap_puro_path if save_images else None,
            "csv_path": csv_path,
            "probabilidad": float(prob),
            "centro": (center_x, center_y),
            "bbox": bbox,
            "tamano_zona_activa": float(area_ratio),
            "nivel_urgencia": float(urgency),
            "nivel_urgencia_label": urgency_label
        }

    def process_folder(self, input_folder, output_root="resultados", threshold=0.7, circle_radius=25, save_images=True):
        """
        Recorre todas las imágenes de input_folder (.jpg/.jpeg/.png) y llama process_image.
        Devuelve una lista con los resultados por imagen.
        """
        results = []
        os.makedirs(output_root, exist_ok=True)
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        for fname in sorted(os.listdir(input_folder)):
            if fname.lower().endswith(valid_ext):
                fp = os.path.join(input_folder, fname)
                try:
                    res = self.process_image(fp, output_root=output_root, threshold=threshold, circle_radius=circle_radius, save_images=save_images)
                    results.append(res)
                    print(f"[OK] Procesada: {fname} -> {res['overlay_path']}")
                except Exception as e:
                    print(f"[ERROR] Al procesar {fname}: {e}")

        return results
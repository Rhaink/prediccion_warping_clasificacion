# Etiquetador de imágenes

Herramienta interactiva para anotar puntos clave sobre radiografías de tórax. Vive de forma aislada dentro de `etiquetado/` y se integra con el resto del repositorio únicamente para leer listas de imágenes y para escribir los CSV finales.

## Requisitos
- Python 3.8+ con un entorno virtual activo.
- Dependencias mínimas: `opencv-python`, `numpy`, `pandas`. Opcionalmente instala el resto del `requirements.txt` si vas a usar otros módulos del repositorio.
- Acceso a un entorno gráfico (el anotador abre ventanas de OpenCV; no funciona en servidores headless sin X11/VNC).

Instalación sugerida:

```bash
pip install -r requirements.txt      # instala todas las dependencias
# o solo lo necesario
pip install opencv-python numpy pandas
```

## Estructura rápida

| Archivo                          | Rol principal                                                                 |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `etiquetar_imagenes.py`          | Ensambla la lista de imágenes a partir del CSV de índices y lanza el anotador |
| `ejecutar_etiquetador.sh`        | Helper que exporta `PYTHONPATH` y ejecuta `etiquetar_imagenes.py`             |
| `etiquetador/main.py`            | Clase `ImageAnnotator`; ciclo principal y orquestación                        |
| `etiquetador/gui_manager.py`     | Ventana OpenCV, callbacks del ratón y render de líneas/puntos                  |
| `etiquetador/event_handler.py`   | Atajos de teclado para mover puntos, guardar o terminar                       |
| `etiquetador/models.py`          | Geometría para los 15 puntos y líneas perpendiculares                         |
| `etiquetador/image_processor.py` | Preprocesado para visualizar (resize + CLAHE)                                 |
| `etiquetador/file_manager.py`    | Escala anotaciones y escribe CSV de salida (64/128/256 px)                    |
| `etiquetador/config.py`          | Constantes de GUI (ventana, colores, teclas y menú impreso)                   |

## Flujo de trabajo
1. `etiquetar_imagenes.py` lee `indices/indices_nuevas_500.csv` (formato: índice nuevo, categoría, id de imagen) y construye rutas dentro de `COVID-19_Radiography_Dataset/<clase>/images/`.
2. La lista de rutas se entrega a `ImageAnnotator(image_paths, archivo_coordenadas_base)`.
3. El anotador muestra cada imagen (redimensionada a 640 px de ancho), espera:
   - Clic izquierdo: primer punto del eje principal.
   - Scroll (EVENT_MOUSEWHEEL): segundo punto.
   - Clic derecho: genera el resto de los 15 puntos.  
   Usa las teclas definidas en `config.py` para ajustes finos, `p` para previsualizar, `l` para limpiar, `s` o `x` para avanzar, `t` para salir.
4. `FileManager` convierte las coordenadas a 64x64, 128x128 y 256x256 y escribe tres archivos `coordenadas/coordenadas_nuevas_500_<res>.csv`. Cada fila: índice original, 30 columnas (x,y) y el nombre de la imagen.

## Cómo ejecutarlo

```bash
cd /ruta/a/Tesis
bash etiquetado/ejecutar_etiquetador.sh
```

El script establece `PYTHONPATH` al directorio padre para que `from etiquetador.main import ImageAnnotator` funcione sin instalación previa. Si ya empaquetaste el módulo (`pip install -e .`) basta con ejecutar `python etiquetado/etiquetar_imagenes.py`.

### Personalización mínima
- Cambia `archivo_indices`, `archivo_coordenadas_base` o `dataset_base_path` en `etiquetar_imagenes.py` según tus carpetas.
- Ajusta `FileManager.output_resolutions` si necesitas resoluciones distintas.
- Modifica `config.py` para cambiar hotkeys, nombre de ventana o colores.
- Si tus imágenes no siguen los patrones `COVID-{id}.png`, etc., actualiza la lógica de nombres en `etiquetar_imagenes.py`.

## Portabilidad
El paquete `etiquetador` no depende del resto del repositorio. Para moverlo a otro proyecto:
1. Copia `etiquetado/` completo o instala el paquete como módulo.
2. Provee tus propias listas de imágenes (CSV, BD, etc.) y llama a `ImageAnnotator`.
3. Reemplaza los caminos rígidos por argumentos o variables de entorno (recomendado añadir `argparse`).
4. Asegúrate de mantener coherente la resolución de visualización (640 px por defecto) con `FileManager.display_resolution`; si la cambias también ajusta el escalado.

## Problemas comunes
- **ImportError de `etiquetador`**: ejecuta desde la raíz del repo o instala el paquete. El shell script ya exporta el `PYTHONPATH` correcto.
- **Sin interfaz gráfica**: OpenCV necesita acceso a DISPLAY/X11. Usa `ssh -X` o un entorno local.
- **CSV de índices vacío**: el script aborta si no encuentra IDs numéricos; revisa el archivo y el dtype usado en `pandas`.
- **Imágenes faltantes**: se imprimen advertencias y se omiten; revisa que los nombres coincidan con el dataset original.

Con esto tienes una visión completa del módulo de etiquetado sin depender del resto de la tesis.

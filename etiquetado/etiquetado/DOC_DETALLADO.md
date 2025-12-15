# Documentación exhaustiva del módulo de etiquetado

Esta referencia describe, línea por línea, todo el código dentro de `etiquetado/`. Se sigue el formato `archivo:línea → explicación`.

## Shell script

- `etiquetado/ejecutar_etiquetador.sh:1` declara el intérprete bash.
- `:3-4` comentario describiendo propósito del script.
- `:6-7` define `DIR` evaluando la ruta absoluta que contiene el propio script (`BASH_SOURCE[0]`).
- `:9-10` exporta `PYTHONPATH` al directorio padre (`Tesis`) para que `import etiquetador` funcione sin instalación.
- `:12-13` ejecuta el script Python principal mediante `python3 "$DIR/etiquetar_imagenes.py"`.

## Script principal (`etiquetar_imagenes.py`)

- `:1` shebang para ejecutar con Python 3.
- `:2-5` docstring que resume el objetivo del archivo.
- `:6-9` importa `os`, `sys`, `pandas` y `numpy`, usados para rutas y manipulación de datos.
- `:11-19` comentarios explicando los ajustes de `sys.path`.
- `:13-15` calculan `SCRIPT_DIR` (directorio actual) y `PROJECT_ROOT` (padre).
- `:20-26` bloque `try/except` que intenta `from etiquetador.main import ImageAnnotator`; si falla imprime mensajes y sale con código 1.
- `:29-124` define `main()`.
- `:32` fija `base_dir` como `PROJECT_ROOT`.
- `:34-40` configura rutas por defecto: CSV de índices y prefijo para los archivos de coordenadas.
- `:42-43` imprime rutas detectadas para diagnóstico.
- `:45-63` bloque `try/except` para leer el CSV mediante `pd.read_csv`. Usa `dtype` explícito (columna 2 como objeto), convierte a entero con `pd.to_numeric`, descarta filas inválidas y reporta cuántas se eliminaron. Si el archivo no existe (57-60) o hay otro error (61-63), aborta.
- `:65` convierte el DataFrame a `numpy` para iterar más rápido.
- `:67-73` determina cuántas imágenes se procesarán y aborta si la lista está vacía.
- `:76-104` recorre los índices, arma el `dataset_base_path` y añade rutas dependiendo de la categoría (`1` COVID, `2` Normal, `3` Viral Pneumonia). Verifica existencia de cada archivo y omite los faltantes.
- `:106-108` valida que haya rutas válidas.
- `:110-117` imprime resumen y describe los tres CSV de salida.
- `:119-122` crea `ImageAnnotator` con la lista y el prefijo y lanza `run()`.
- `:125-126` ejecuta `main()` cuando el script se invoca directamente.

## Paquete `etiquetador`

### `__init__.py`
- `:1-3` docstring general.
- `:4-9` importa y expone las clases principales (`ImageAnnotator`, `Point`, `Line`, etc.).
- `:11-20` define `__all__` con los símbolos exportados.

### `config.py`
- `:1-3` docstring.
- `:5-8` fijan nombre y tamaño de la ventana principal (640×640).
- `:10-13` definen colores BGR para líneas, puntos y línea central.
- `:15-41` mapa `TECLAS` con todos los atajos del teclado.
- `:44-51` diccionario `MENSAJES` para los textos impresos en consola.
- `:53-81` literal de cadena `MENU_TEXTO` con el menú que se imprime al iniciar cada imagen.

### `main.py`
- `:1-3` docstring que describe el archivo.
- `:4-11` importa `cv2`, `numpy` y dependencias internas.
- `:13-91` clase `ImageAnnotator`.
- `:16-34` `__init__`: guarda rutas, índice actual, crea `GUIManager`, `EventHandler`, `FileManager`, `ImageProcessor` y registra el `mouse_callback` con `cv2.setMouseCallback`.
- `:36-92` método `run`: bucle principal. Para cada imagen imprime la ruta (`:40-43`), notifica al `FileManager` (`:45`), lee la imagen con `cv2.imread` y salta si falla (`:47-52`), reinicia GUI y handler (`:54-58`), procesa y muestra imagen (`:60-67`), entra en bucle `while True` donde actualiza la ventana (`:71-78`), delega teclas a `EventHandler`. Cuando obtiene `'break'`, recupera la anotación (`:82-85`), guarda si está completa y avanza. Si recibe `'exit'`, corta el programa.
- `:91-92` imprime mensaje final y cierra ventanas.

### `gui_manager.py`
- `:1-17` docstring e importaciones.
- `:19-199` clase `GUIManager`.
- `:22-30` `__init__` inicializa atributos y llama a `_setup_window`.
- `:32-34` `_setup_window` crea la ventana configurable con `cv2.namedWindow`.
- `:36-58` `set_image` copia la imagen, calcula grosor, redimensiona la ventana y dibuja la línea central vertical.
- `:59-97` `mouse_callback` responde a eventos: clic izquierdo define punto 1, rueda define punto 2 y dibuja la línea principal, clic derecho calcula todos los puntos restantes (invoca `ImageAnnotation.calculate_all_points`) y los dibuja.
- `:98-157` métodos privados `_draw_point`, `_draw_line`, `_draw_main_line`, `_calculate_perpendicular_points` (mantiene compatibilidad con la versión anterior), `_draw_all_elements` que limpia la imagen y redibuja línea central, eje principal, perpendiculares (pares `[2,3]`, `[4,5]`, etc.) y resalta los puntos centrales (8-10).
- `:160-177` `show_preview` muestra ventana adicional con los puntos numerados si todos existen.
- `:178-184` `clear_image` restaura la imagen original más la línea central.
- `:185-188` `show_menu` imprime `MENU_TEXTO`.
- `:189-194` `update_display` llama a `cv2.imshow` y silencia excepciones (p. ej. ventana cerrada manualmente).
- `:196-198` `close_windows`.
- `:200-203` `get_annotation` devuelve la instancia `ImageAnnotation`.

### `event_handler.py`
- `:1-9` docstring e importaciones.
- `:10-84` clase `EventHandler`.
- `:13-21` constructor guarda referencias a `GUIManager` y a la anotación actual.
- `:23-72` `handle_keyboard_event`: recibe `key` (código ASCII), lo transforma a carácter (`chr(key & 0xFF)`) y compara contra `TECLAS`. Por cada tecla de movimiento invoca `_mover_punto` con un índice y dirección (-1 izquierda, 1 derecha). Las teclas especiales llaman a `GUIManager` (`p`, `l`), imprimen mensaje y retornan `'break'` (`s`, `x`) o `'exit'` (`t`). El resto devuelve `'continue'`.
- `:74-84` `_mover_punto` llama `ImageAnnotation.move_point`, redibuja todo y actualiza la ventana.

### `file_manager.py`
- `:1-7` docstring e importaciones.
- `:8-115` clase `FileManager`.
- `:11-22` constructor recibe la ruta base (prefijo) del CSV, inicializa índice actual, nombre de imagen y define resoluciones objetivo `[64, 128, 256]`.
- `:24-34` `set_current_image` extrae el nombre del archivo sin extensión (dividiendo por `'/'` y `'.'`) y lo guarda junto con el índice.
- `:35-81` `_scale_to_resolution`: escala cada `Point` desde la resolución de visualización (asumida 640 px) a la resolución deseada. Calcula `scale_factor`, centro de masa de todos los puntos y usa traslación relativa (`dx`, `dy`) para preservar relaciones geométricas; redondea y acota valores entre 0 y `target_resolution - 1`.
- `:83-95` `_get_output_filename` arma el nombre final sumando `_64x64.csv`, etc., al prefijo.
- `:96-115` `save_annotation`: ignora anotaciones incompletas, itera sobre cada resolución, escala puntos y escribe una fila en el CSV (`csv.writer`) con `[current_index] + coords + [current_image]`.

### `image_processor.py`
- `:1-6` docstring e importaciones.
- `:8-89` clase `ImageProcessor`.
- `:11-25` `process_image` redimensiona manteniendo proporción a `VENTANA_ANCHO` y aplica CLAHE para mejorar contraste.
- `:27-55` `resize_aspect_ratio` calcula dimensiones usando el ancho o alto objetivo y devuelve `cv2.resize`.
- `:56-76` `apply_clahe` convierte la imagen a LAB, aplica CLAHE al canal L y vuelve a BGR.
- `:78-89` `calculate_line_thickness` determina grosor en función del tamaño (mínimo de alto/ancho dividido entre 200).

### `models.py`
- `:1-7` docstring e importaciones.
- `:9-38` clase `Point`: representa coordenadas, permite convertir a tupla, escalar (`scale`) y mover horizontalmente (`move_horizontal`) respetando una pendiente para recalcular `y`.
- `:39-88` clase `Line`: calcula pendiente y perpendicular, obtiene puntos a una fracción (`get_point_at_fraction`) o pares perpendiculares (`get_perpendicular_points`).
- `:90-240` clase `ImageAnnotation`: núcleo geométrico.
  - `:93-97` inicializa lista de 15 puntos, línea principal y diccionario de puntos intermedios.
  - `:98-107` `set_point` y `get_point`.
  - `:109-144` `move_point`: usa `intermediate_points` y la pendiente de la línea principal para desplazar horizontalmente los puntos 3-8 y 12-15. Recalcula `y` con `Point.move_horizontal`.
  - `:145-147` `are_all_points_defined`.
  - `:149-153` `calculate_main_line`.
  - `:155-233` `calculate_all_points`: si los puntos 1 y 2 no son verticales, calcula la pendiente, puntos medios y cuartos, guarda referencias en `intermediate_points` y crea líneas perpendiculares mediante `create_perpendicular_line`. Asigna los puntos resultantes 3-15; si la línea es vertical (`:234-240`), usa un camino alterno con valores fijos para `x`.

## Uso de constantes y eventos

- Todas las claves del teclado, mensajes y texto del menú provienen de `config.py`. Cada que se imprime algo en consola se referencian esas constantes (por ejemplo, `MENSAJES['PRIMER_CLICK']` en `gui_manager.py:75`).
- El comportamiento del mouse está definido explícitamente: clic izquierdo, rueda (OpenCV la expone como `EVENT_MOUSEWHEEL`) y clic derecho.
- Los prints dentro de `main.py` y `etiquetar_imagenes.py` ayudan a rastrear progreso y errores; se documentaron en las secciones anteriores.

Con esta guía, cada línea del código queda explicada y se puede rastrear rápidamente qué hace cada función dentro del proyecto de etiquetado.

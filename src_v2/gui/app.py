"""
Gradio web interface for COVID-19 detection demonstration.

Provides three tabs:
1. Full demo: Complete pipeline visualization
2. Quick view: Fast classification only
3. About: System information
"""
import gradio as gr
import pandas as pd

from .inference_pipeline import (
    process_image_full,
    process_image_quick,
    export_results,
)
from .config import (
    TITLE,
    SUBTITLE,
    AUTHOR_FOOTER,
    ABOUT_TEXT,
    THEME,
    populate_examples,
    VALIDATED_METRICS,
    get_class_color_es,
)
from .visualizer import create_probability_chart


def create_probability_html(probabilities: dict, predicted_class: str) -> str:
    """
    Crea HTML personalizado para mostrar probabilidades con colores y emoji.

    Args:
        probabilities: Diccionario {clase: probabilidad}
        predicted_class: Nombre de la clase ganadora

    Returns:
        String HTML con barras de progreso coloreadas
    """
    # Ordenar por probabilidad descendente
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    html = """
    <div style="font-family: 'Inter', sans-serif; padding: 10px;">
    """

    for class_name, prob in sorted_probs:
        display_name = class_name  # Usar nombre completo directamente
        percentage = prob * 100
        color = get_class_color_es(class_name)

        # Agregar emoji solo a la ganadora
        if class_name == predicted_class:
            display_name = f"⭐ {display_name}"
            font_weight = "bold"
        else:
            font_weight = "normal"

        html += f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: {font_weight}; color: {color}; font-size: 14px;">{display_name}</span>
                <span style="font-weight: {font_weight}; color: {color}; font-size: 14px;">{percentage:.1f}%</span>
            </div>
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 8px; overflow: hidden;">
                <div style="background-color: {color}; height: 100%; width: {percentage:.1f}%; border-radius: 10px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """

    html += "</div>"
    return html


def create_prediction_display_html(predicted_class: str, confidence: float) -> str:
    """
    Crea HTML para mostrar la clase predicha de forma destacada.

    Args:
        predicted_class: Nombre de la clase predicha
        confidence: Confianza de la predicción (0-1)

    Returns:
        HTML con la clase predicha en grande y con color
    """
    color = get_class_color_es(predicted_class)
    percentage = confidence * 100

    html = f"""
    <div style="text-align: center; padding: 15px; background-color: #2a2a2a; border-radius: 10px; margin-bottom: 15px;">
        <div style="font-size: 32px; font-weight: bold; color: {color};">
            ⭐ {predicted_class}
        </div>
    </div>
    """
    return html


def highlight_winner_in_probabilities(probabilities: dict, predicted_class: str) -> dict:
    """
    Crea una copia del diccionario de probabilidades con la clase ganadora resaltada.
    Redondea a 1 decimal para consistencia.

    Args:
        probabilities: Diccionario original {clase: probabilidad}
        predicted_class: Nombre de la clase ganadora

    Returns:
        Diccionario con la clase ganadora resaltada con ⭐
    """
    highlighted = {}
    for class_name, prob in probabilities.items():
        # Usar nombre completo directamente
        display_name = class_name

        # Redondear a 1 decimal para consistencia
        prob_rounded = round(prob, 3)  # Mantener 3 decimales internos para precisión

        if class_name == predicted_class:
            # Agregar emoji y color para la clase ganadora
            class_color = get_class_color_es(class_name)
            highlighted[f"⭐ {display_name}"] = prob_rounded
        else:
            highlighted[display_name] = prob_rounded

    return highlighted


def create_demo() -> gr.Blocks:
    """
    Create Gradio Blocks interface.

    Returns:
        Gradio Blocks app
    """
    # Populate examples
    examples = populate_examples()

    # Theme
    if THEME == "soft":
        theme = gr.themes.Soft()
    elif THEME == "glass":
        theme = gr.themes.Glass()
    elif THEME == "monochrome":
        theme = gr.themes.Monochrome()
    else:
        theme = gr.themes.Default()

    with gr.Blocks(theme=theme, title=TITLE) as demo:
        # Header
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(SUBTITLE)
        gr.HTML(AUTHOR_FOOTER)

        with gr.Tabs():
            # ================================================================
            # TAB 1: FULL DEMO
            # ================================================================
            with gr.TabItem("📊 Demostración Completa"):
                gr.Markdown("""
                ### Pipeline Completo
                Este modo muestra las 4 etapas del sistema:
                1. **Imagen Original** → 2. **Puntos de Referencia Detectados** → 3. **Imagen Normalizada** → 4. **SAHS (Mejora de Contraste)**
                """)

                with gr.Row():
                    # Left column: Input
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="filepath",
                            label="Cargar Radiografía de Tórax",
                            height=300
                        )

                        process_btn = gr.Button(
                            "🔍 Procesar Imagen",
                            variant="primary",
                            size="lg"
                        )

                        # Examples
                        if examples:
                            gr.Examples(
                                examples=[[ex[0]] for ex in examples],
                                inputs=input_image,
                                label="Ejemplos Precargados"
                            )

                        # Export button (initially hidden)
                        export_btn = gr.Button(
                            "💾 Exportar Resultados a PDF",
                            variant="secondary",
                            visible=True
                        )

                        export_status = gr.Textbox(
                            label="Estado de Exportación",
                            interactive=False,
                            visible=False
                        )

                    # Right column: Outputs
                    with gr.Column(scale=2):
                        # Processing status
                        status_text = gr.Markdown("Esperando imagen...")

                        # Step-by-step visualizations
                        with gr.Row():
                            img_original = gr.Image(
                                label="1️⃣ Imagen Original",
                                type="pil",
                                height=300
                            )
                            img_landmarks = gr.Image(
                                label="2️⃣ Puntos de Referencia Detectados (15 puntos)",
                                type="pil",
                                height=300
                            )

                        with gr.Row():
                            img_delaunay = gr.Image(
                                label="🔷 Malla de Delaunay",
                                type="pil",
                                height=300
                            )
                            img_warped = gr.Image(
                                label="3️⃣ Imagen Normalizada (Warped)",
                                type="pil",
                                height=300
                            )

                        with gr.Row():
                            img_sahs = gr.Image(
                                label="4️⃣ Imagen Normalizada con SAHS",
                                type="pil",
                                height=300
                            )

                        # Classification results
                        gr.Markdown("### Resultados de Clasificación")

                        # Mostrar predicción destacada
                        predicted_class_display = gr.HTML(
                            label="Predicción"
                        )

                        # Barras de probabilidades
                        classification_html = gr.HTML(
                            label="Probabilidades por Clase"
                        )

                        # Metrics accordion
                        with gr.Accordion("📈 Métricas Detalladas", open=False):
                            metrics_table = gr.Dataframe(
                                label="Coordenadas de Puntos de Referencia Detectados",
                                interactive=False
                            )

                            inference_time = gr.Textbox(
                                label="Tiempo de Inferencia",
                                interactive=False
                            )

                # Hidden state to store results for export
                result_state = gr.State(value=None)

                # Process button click
                def on_process(image_path):
                    if image_path is None:
                        return (
                            "⚠️ Por favor, cargue una imagen primero.",
                            None, None, None, None, None, None, None, None, None, None
                        )

                    # Process image
                    result = process_image_full(image_path)

                    if not result['success']:
                        error_msg = f"❌ **Error**: {result['error']}"
                        return (
                            error_msg,
                            None, None, None, None, None, None, None, None, None,
                            result  # Store result for potential export
                        )

                    # Success
                    status_msg = f"✅ **Procesamiento completado en {result['inference_time']:.2f} segundos**"

                    if result.get('warping_failed', False):
                        status_msg += "\n⚠️ Advertencia: Warping falló, mostrando imagen original."

                    # Add prediction with emoji and color
                    predicted_class = result['predicted_class']
                    predicted_prob = result['classification'][predicted_class]  # 0-1
                    class_color = get_class_color_es(predicted_class)

                    status_msg += f'\n\n### <span style="color: {class_color};">⭐ {predicted_class}</span>\n\n**Confianza**: {predicted_prob * 100:.1f}%'

                    # Create HTML de predicción destacada
                    prediction_display_html = create_prediction_display_html(
                        predicted_class,
                        predicted_prob
                    )

                    # Create HTML for probabilities display with colors
                    probabilities_html = create_probability_html(
                        result['classification'],
                        predicted_class
                    )

                    return (
                        status_msg,
                        result['original'],
                        result['landmarks'],
                        result['delaunay_mesh'],
                        result['warped'],
                        result['warped_sahs'],
                        prediction_display_html,  # Nuevo output
                        probabilities_html,        # Barras
                        result['metrics'],
                        f"{result['inference_time']:.3f} segundos",
                        result  # Store for export
                    )

                process_btn.click(
                    fn=on_process,
                    inputs=[input_image],
                    outputs=[
                        status_text,
                        img_original,
                        img_landmarks,
                        img_delaunay,
                        img_warped,
                        img_sahs,
                        predicted_class_display,  # Nuevo
                        classification_html,       # Barras
                        metrics_table,
                        inference_time,
                        result_state
                    ]
                )

                # Export button click
                def on_export(result):
                    if result is None or not result.get('success', False):
                        return "⚠️ No hay resultados válidos para exportar.", True

                    success, message = export_results(result)
                    return message, True

                export_btn.click(
                    fn=on_export,
                    inputs=[result_state],
                    outputs=[export_status, export_status]  # Show status component
                )

            # ================================================================
            # TAB 2: QUICK VIEW
            # ================================================================
            with gr.TabItem("⚡ Vista Rápida"):
                gr.Markdown("""
                ### Clasificación Rápida
                Este modo realiza solo la clasificación, sin visualizaciones intermedias.
                Ideal para procesar múltiples imágenes rápidamente.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        quick_input = gr.Image(
                            type="filepath",
                            label="Cargar Radiografía de Tórax",
                            height=400
                        )

                        quick_btn = gr.Button(
                            "🚀 Clasificar",
                            variant="primary",
                            size="lg"
                        )

                        # Examples
                        if examples:
                            gr.Examples(
                                examples=[[ex[0]] for ex in examples],
                                inputs=quick_input,
                                label="Ejemplos Precargados"
                            )

                    with gr.Column(scale=1):
                        quick_status = gr.Markdown("Esperando imagen...")

                        # Mostrar predicción destacada
                        quick_predicted_display = gr.HTML(
                            label="Predicción"
                        )

                        # Barras de probabilidades
                        quick_output = gr.HTML(
                            label="Probabilidades por Clase"
                        )

                        quick_time = gr.Textbox(
                            label="Tiempo de Inferencia",
                            interactive=False
                        )

                # Quick classification
                def on_quick_classify(image_path):
                    if image_path is None:
                        return "⚠️ Por favor, cargue una imagen primero.", None, None, ""

                    result = process_image_quick(image_path)

                    if not result['success']:
                        return f"❌ **Error**: {result['error']}", None, None, ""

                    # Add prediction with emoji and color
                    predicted_class = result['predicted_class']
                    predicted_prob = result['classification'][predicted_class]
                    class_color = get_class_color_es(predicted_class)

                    status_msg = f'✅ **Clasificación completada**\n\n### <span style="color: {class_color};">⭐ {predicted_class}</span>'

                    # Create HTML de predicción destacada
                    prediction_display_html = create_prediction_display_html(
                        predicted_class,
                        predicted_prob
                    )

                    # Create HTML for probabilities display with colors
                    probabilities_html = create_probability_html(
                        result['classification'],
                        predicted_class
                    )

                    return (
                        status_msg,
                        prediction_display_html,  # Nuevo output
                        probabilities_html,        # Barras
                        f"{result['inference_time']:.3f} segundos"
                    )

                quick_btn.click(
                    fn=on_quick_classify,
                    inputs=[quick_input],
                    outputs=[
                        quick_status,
                        quick_predicted_display,  # Nuevo
                        quick_output,              # Barras
                        quick_time
                    ]
                )

            # ================================================================
            # TAB 3: ABOUT (oculto por ahora)
            # ================================================================
            # with gr.TabItem("ℹ️ Acerca del Sistema"):
            #     gr.Markdown(ABOUT_TEXT)
            #
            #     # Footer with metrics
            #     gr.Markdown("---")
            #     gr.Markdown(f"""
            #     ### Métricas Validadas
            #
            #     | Métrica | Valor |
            #     |---------|-------|
            #     | Error de Landmarks (Ensemble) | {VALIDATED_METRICS['landmark_error_px']:.2f} ± {VALIDATED_METRICS['landmark_std_px']:.2f} px |
            #     | Mediana de Error | {VALIDATED_METRICS['landmark_median_px']:.2f} px |
            #     | Accuracy de Clasificación | {VALIDATED_METRICS['classification_accuracy']:.2f}% |
            #     | F1-Score Macro | {VALIDATED_METRICS['classification_f1_macro']:.2f}% |
            #     | F1-Score Weighted | {VALIDATED_METRICS['classification_f1_weighted']:.2f}% |
            #     | Tamaño de Imagen | {VALIDATED_METRICS['model_input_size']}×{VALIDATED_METRICS['model_input_size']} px |
            #     | Fill Rate | {VALIDATED_METRICS['fill_rate']}% |
            #
            #     **Preprocesamiento:**
            #     - CLAHE: clip={VALIDATED_METRICS['clahe_clip']}, tile={VALIDATED_METRICS['clahe_tile']}×{VALIDATED_METRICS['clahe_tile']}
            #     - Margen de Warping: {VALIDATED_METRICS['margin_scale']}× desde centroide
            #     """)

    return demo


if __name__ == "__main__":
    # For testing
    demo = create_demo()
    demo.launch(server_name="localhost", server_port=7860)

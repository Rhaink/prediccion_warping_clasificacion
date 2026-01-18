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
    ABOUT_TEXT,
    THEME,
    populate_examples,
    VALIDATED_METRICS,
)
from .visualizer import create_probability_chart


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

        with gr.Tabs():
            # ================================================================
            # TAB 1: FULL DEMO
            # ================================================================
            with gr.TabItem("📊 Demostración Completa"):
                gr.Markdown("""
                ### Pipeline Completo
                Este modo muestra las 4 etapas del sistema:
                1. **Imagen Original** → 2. **Landmarks Detectados** → 3. **Imagen Normalizada** → 4. **GradCAM (Explicabilidad)**
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
                                label="2️⃣ Landmarks Detectados (15 puntos)",
                                type="pil",
                                height=300
                            )

                        with gr.Row():
                            img_warped = gr.Image(
                                label="3️⃣ Imagen Normalizada (Warped)",
                                type="pil",
                                height=300
                            )
                            img_gradcam = gr.Image(
                                label="4️⃣ GradCAM: Regiones de Atención",
                                type="pil",
                                height=300
                            )

                        # Classification results
                        gr.Markdown("### Resultados de Clasificación")
                        classification_label = gr.Label(
                            label="Probabilidades por Clase",
                            num_top_classes=3
                        )

                        # Metrics accordion
                        with gr.Accordion("📈 Métricas Detalladas", open=False):
                            metrics_table = gr.Dataframe(
                                label="Error por Landmark (Valores de Referencia)",
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
                            None, None, None, None, None, None, None, None
                        )

                    # Process image
                    result = process_image_full(image_path)

                    if not result['success']:
                        error_msg = f"❌ **Error**: {result['error']}"
                        return (
                            error_msg,
                            None, None, None, None, None, None, None,
                            result  # Store result for potential export
                        )

                    # Success
                    status_msg = f"✅ **Procesamiento completado en {result['inference_time']:.2f} segundos**"

                    if result.get('warping_failed', False):
                        status_msg += "\n⚠️ Advertencia: Warping falló, mostrando imagen original."

                    status_msg += f"\n\n**Predicción**: {result['predicted_class']} ({result['classification'][result['predicted_class']] * 100:.1f}% confianza)"

                    return (
                        status_msg,
                        result['original'],
                        result['landmarks'],
                        result['warped'],
                        result['gradcam'],
                        result['classification'],
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
                        img_warped,
                        img_gradcam,
                        classification_label,
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

                        quick_output = gr.Label(
                            label="Resultado de Clasificación",
                            num_top_classes=3
                        )

                        quick_time = gr.Textbox(
                            label="Tiempo de Inferencia",
                            interactive=False
                        )

                # Quick classification
                def on_quick_classify(image_path):
                    if image_path is None:
                        return "⚠️ Por favor, cargue una imagen primero.", None, ""

                    result = process_image_quick(image_path)

                    if not result['success']:
                        return f"❌ **Error**: {result['error']}", None, ""

                    status_msg = f"✅ **Clasificación completada**\n\n**Predicción**: {result['predicted_class']}"

                    return (
                        status_msg,
                        result['classification'],
                        f"{result['inference_time']:.3f} segundos"
                    )

                quick_btn.click(
                    fn=on_quick_classify,
                    inputs=[quick_input],
                    outputs=[quick_status, quick_output, quick_time]
                )

            # ================================================================
            # TAB 3: ABOUT
            # ================================================================
            with gr.TabItem("ℹ️ Acerca del Sistema"):
                gr.Markdown(ABOUT_TEXT)

                # Footer with metrics
                gr.Markdown("---")
                gr.Markdown(f"""
                ### Métricas Validadas

                | Métrica | Valor |
                |---------|-------|
                | Error de Landmarks (Ensemble) | {VALIDATED_METRICS['landmark_error_px']:.2f} ± {VALIDATED_METRICS['landmark_std_px']:.2f} px |
                | Mediana de Error | {VALIDATED_METRICS['landmark_median_px']:.2f} px |
                | Accuracy de Clasificación | {VALIDATED_METRICS['classification_accuracy']:.2f}% |
                | F1-Score Macro | {VALIDATED_METRICS['classification_f1_macro']:.2f}% |
                | F1-Score Weighted | {VALIDATED_METRICS['classification_f1_weighted']:.2f}% |
                | Tamaño de Imagen | {VALIDATED_METRICS['model_input_size']}×{VALIDATED_METRICS['model_input_size']} px |
                | Fill Rate | {VALIDATED_METRICS['fill_rate']}% |

                **Preprocesamiento:**
                - CLAHE: clip={VALIDATED_METRICS['clahe_clip']}, tile={VALIDATED_METRICS['clahe_tile']}×{VALIDATED_METRICS['clahe_tile']}
                - Margen de Warping: {VALIDATED_METRICS['margin_scale']}× desde centroide
                """)

    return demo


if __name__ == "__main__":
    # For testing
    demo = create_demo()
    demo.launch(server_name="localhost", server_port=7860)

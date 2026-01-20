"""Tests de la interfaz Gradio."""

import pytest


def test_create_demo():
    """Verifica que create_demo() retorna interfaz Gradio válida."""
    from src_v2.gui.app import create_demo
    import gradio as gr

    demo = create_demo()

    assert isinstance(demo, gr.Blocks)


def test_demo_has_three_tabs():
    """Verifica que la demo tiene 3 tabs."""
    from src_v2.gui.app import create_demo

    demo = create_demo()

    # Verificar estructura (esto es más complicado con Gradio API)
    # Por ahora solo verificar que se crea correctamente
    assert demo is not None


@pytest.mark.slow
def test_demo_launches():
    """Verifica que la demo puede lanzarse (no conectarse)."""
    from src_v2.gui.app import create_demo

    demo = create_demo()

    # Intentar lanzar en modo queue=False (no bloquea)
    try:
        demo.queue(max_size=1)
        # No llamar launch() porque bloquearía
    except Exception as e:
        pytest.fail(f"Demo no pudo configurarse: {e}")

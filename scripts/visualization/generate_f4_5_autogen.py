#!/usr/bin/env python3
"""Generate an auto diagram for the F4.5 landmark model architecture.

This script produces a diagram from the PyTorch model definition using one of:
- torchview (layer graph, cleaner)
- torchviz (autograd graph, very detailed)
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from src_v2.models.resnet_landmark import ResNet18Landmarks


LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/final_config.json")
DEFAULT_OUTPUT = Path(
    "outputs/thesis_figures_final/cap4_metodologia/F4.5_arquitectura_modelo_autogen.png"
)
DEFAULT_OUTPUT_SUMMARY = Path(
    "outputs/thesis_figures_final/cap4_metodologia/F4.5_arquitectura_modelo_autogen_summary.png"
)
DEFAULT_INPUT_HW = (224, 224)
DEFAULT_FIGSIZE = (12.0, 4.0)


def _filter_kwargs(func: object, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return kwargs supported by the callable."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    allowed = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def _load_arch_config(config_path: Path) -> Dict[str, Any]:
    """Load architecture settings from a JSON config."""
    if not config_path.exists():
        LOGGER.warning("Config not found: %s. Using defaults.", config_path)
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("architecture", {})


def _build_model(
    arch_config: Dict[str, Any],
    pretrained: bool,
    device: torch.device,
) -> ResNet18Landmarks:
    """Create the landmarks model with optional overrides."""
    use_coord_attention = bool(arch_config.get("use_coord_attention", True))
    deep_head = bool(arch_config.get("deep_head", True))
    hidden_dim = int(arch_config.get("hidden_dim", 768))
    dropout_rate = float(arch_config.get("dropout_rate", 0.3))
    num_landmarks = int(arch_config.get("num_landmarks", 15))

    model = ResNet18Landmarks(
        num_landmarks=num_landmarks,
        pretrained=pretrained,
        freeze_backbone=True,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        use_coord_attention=use_coord_attention,
        deep_head=deep_head,
    )
    model.eval()
    return model.to(device)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _render_with_torchview(
    model: ResNet18Landmarks,
    input_size: Tuple[int, int, int, int],
    output_path: Path,
    fmt: str,
    device: str,
) -> Path:
    try:
        from torchview import draw_graph
    except ImportError as exc:
        raise RuntimeError("torchview is not installed") from exc

    draw_kwargs = _filter_kwargs(
        draw_graph,
        {
            "input_size": input_size,
            "device": device,
            "expand_nested": True,
        },
    )
    graph = draw_graph(model, **draw_kwargs)
    visual_graph = graph.visual_graph
    visual_graph.attr(rankdir="LR")

    _ensure_parent(output_path)
    render_path = output_path.with_suffix("")
    try:
        visual_graph.render(str(render_path), format=fmt, cleanup=True)
    except Exception as exc:
        raise RuntimeError(
            "Graphviz rendering failed for torchview output. "
            "Ensure Graphviz is installed and on PATH."
        ) from exc

    return output_path


def _extract_tensor_shape(value: object) -> Tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(int(x) for x in value.shape)
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return tuple(int(x) for x in item.shape)
    return None


def _collect_stage_shapes(
    model: ResNet18Landmarks,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    hooks = []

    def register(module: nn.Module, name: str) -> None:
        def _hook(_module: nn.Module, _inputs: tuple, outputs: object) -> None:
            shape = _extract_tensor_shape(outputs)
            if shape is not None:
                shapes[name] = shape

        hooks.append(module.register_forward_hook(_hook))

    register(model.backbone_conv, "backbone")
    if model.coord_attention is not None:
        register(model.coord_attention, "coord_attention")
    register(model.avgpool, "avgpool")
    register(model.head, "head")

    inputs = torch.zeros(*input_size, device=device)
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()

    return shapes


def _format_feature_shape(shape: Tuple[int, ...] | None) -> str:
    if not shape:
        return "?"
    if len(shape) == 4:
        _, channels, height, width = shape
        return f"{height}x{width}x{channels}"
    if len(shape) == 2:
        return str(shape[1])
    return "x".join(str(dim) for dim in shape[1:])


def _extract_linear_dims(head: nn.Module) -> Tuple[int, ...]:
    dims = []
    for layer in head.modules():
        if isinstance(layer, nn.Linear):
            if not dims:
                dims.append(layer.in_features)
            dims.append(layer.out_features)
    if not dims:
        return ()
    simplified = [dims[0]]
    for dim in dims[1:]:
        if dim != simplified[-1]:
            simplified.append(dim)
    return tuple(simplified)


def _render_summary_matplotlib(
    model: ResNet18Landmarks,
    input_hw: Tuple[int, int],
    output_path: Path,
    device: torch.device,
    figsize: Tuple[float, float],
    font_scale: float,
    grayscale: bool,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for summary diagrams") from exc

    input_size = (1, 3, input_hw[0], input_hw[1])
    shapes = _collect_stage_shapes(model, input_size, device)
    head_dims = _extract_linear_dims(model.head)
    head_label = " -> ".join(str(dim) for dim in head_dims) if head_dims else "?"

    if grayscale:
        colors = {
            "input": "#E0E0E0",
            "backbone": "#D6D6D6",
            "attention": "#CBCBCB",
            "head": "#E6E6E6",
            "output": "#F0F0F0",
            "border": "#303030",
            "text": "#202020",
            "dim": "#606060",
        }
    else:
        colors = {
            "input": "#B0BEC5",
            "backbone": "#90CAF9",
            "attention": "#80DEEA",
            "head": "#FFAB91",
            "output": "#EF9A9A",
            "border": "#37474F",
            "text": "#212121",
            "dim": "#616161",
        }

    blocks = [
        {
            "label": "Entrada",
            "sublabel": f"{input_hw[0]}x{input_hw[1]}x3",
            "color": colors["input"],
            "width": 1.7,
        },
        {
            "label": "ResNet-18",
            "sublabel": _format_feature_shape(shapes.get("backbone")),
            "color": colors["backbone"],
            "width": 2.2,
        },
    ]

    if model.coord_attention is not None:
        blocks.append(
            {
                "label": "Coordinate\nAttention",
                "sublabel": _format_feature_shape(shapes.get("coord_attention")),
                "color": colors["attention"],
                "width": 2.2,
            }
        )

    blocks.append(
        {
            "label": "GAP +\nCabeza",
            "sublabel": head_label,
            "color": colors["head"],
            "width": 2.0,
        }
    )

    label_size = 11 * font_scale
    sublabel_size = 9 * font_scale
    output_size = 12 * font_scale

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    y_main = 2.2
    block_count = len(blocks)
    x_start = 1.0
    x_end = 9.6 if block_count <= 4 else 10.2
    step = (x_end - x_start) / max(block_count - 1, 1)
    x_positions = [x_start + step * idx for idx in range(block_count)]

    def draw_block(x, data):
        width = data["width"]
        height = 1.5
        rect = FancyBboxPatch(
            (x - width / 2, y_main - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=data["color"],
            edgecolor=colors["border"],
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y_main,
            data["label"],
            ha="center",
            va="center",
            fontsize=label_size,
            fontweight="bold",
            color=colors["text"],
            zorder=3,
        )
        if data.get("sublabel"):
            ax.text(
                x,
                y_main - height / 2 - 0.2,
                data["sublabel"],
                ha="center",
                va="top",
                fontsize=sublabel_size,
                color=colors["dim"],
                style="italic",
            )

    def draw_arrow(x1, x2):
        ax.annotate(
            "",
            xy=(x2, y_main),
            xytext=(x1, y_main),
            arrowprops=dict(arrowstyle="->", color=colors["border"], lw=2),
        )

    for idx, block in enumerate(blocks):
        draw_block(x_positions[idx], block)
        if idx < len(blocks) - 1:
            start_x = x_positions[idx] + block["width"] / 2
            end_x = x_positions[idx + 1] - blocks[idx + 1]["width"] / 2
            draw_arrow(start_x, end_x)

    output_x = x_end + 1.2
    draw_arrow(x_positions[-1] + blocks[-1]["width"] / 2, output_x - 0.3)
    ax.text(
        output_x,
        y_main,
        "15 x 2 coords",
        ha="left",
        va="center",
        fontsize=output_size,
        fontweight="bold",
        color=colors["text"],
    )

    plt.tight_layout(pad=0.3)
    _ensure_parent(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


def _render_with_torchviz(
    model: ResNet18Landmarks,
    input_size: Tuple[int, int, int, int],
    output_path: Path,
    fmt: str,
    device: torch.device,
) -> Path:
    try:
        from torchviz import make_dot
    except ImportError as exc:
        raise RuntimeError("torchviz is not installed") from exc

    inputs = torch.randn(*input_size, device=device, requires_grad=True)
    outputs = model(inputs)
    dot = make_dot(outputs.mean(), params=dict(model.named_parameters()))
    dot.format = fmt

    _ensure_parent(output_path)
    render_path = output_path.with_suffix("")
    try:
        dot.render(str(render_path), cleanup=True)
    except Exception as exc:
        raise RuntimeError(
            "Graphviz rendering failed for torchviz output. "
            "Ensure Graphviz is installed and on PATH."
        ) from exc

    return output_path


def _parse_input_hw(values: Iterable[int]) -> Tuple[int, int]:
    values = tuple(values)
    if len(values) != 2:
        raise ValueError("input size must be two integers: H W")
    return int(values[0]), int(values[1])


def generate_diagram(
    backend: str,
    model: ResNet18Landmarks,
    input_hw: Tuple[int, int],
    output_path: Path,
    device: torch.device,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    font_scale: float = 1.0,
    grayscale: bool = False,
) -> Path:
    """Generate the auto diagram using the selected backend."""
    input_size = (1, 3, input_hw[0], input_hw[1])
    fmt = output_path.suffix.lstrip(".") or "png"

    if backend == "torchview":
        return _render_with_torchview(model, input_size, output_path, fmt, str(device))
    if backend == "torchviz":
        return _render_with_torchviz(model, input_size, output_path, fmt, device)
    if backend == "summary":
        return _render_summary_matplotlib(
            model,
            input_hw,
            output_path,
            device,
            figsize,
            font_scale,
            grayscale,
        )

    last_error = None
    for candidate in ("torchview", "torchviz"):
        try:
            return generate_diagram(
                candidate,
                model,
                input_hw,
                output_path,
                device,
                figsize=figsize,
                font_scale=font_scale,
                grayscale=grayscale,
            )
        except RuntimeError as exc:
            last_error = exc
            LOGGER.warning("Backend %s failed: %s", candidate, exc)

    if last_error is not None:
        raise last_error
    raise RuntimeError("No diagram backend is available")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an auto diagram for the F4.5 landmark model architecture."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "torchview", "torchviz", "summary"),
        default="auto",
        help="Diagram backend (default: auto).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the JSON config with architecture settings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path (png/pdf/svg).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=DEFAULT_INPUT_HW,
        metavar=("H", "W"),
        help="Input size (height width).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        help="Matplotlib figure size in inches (summary backend only).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Font size scale for the summary backend.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Use a grayscale palette for the summary backend.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights (may download).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use (default: cpu).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_hw = _parse_input_hw(args.input_size)
    device = torch.device(args.device)
    arch_config = _load_arch_config(args.config)
    model = _build_model(arch_config, pretrained=args.pretrained, device=device)

    output_path = args.output
    if args.output == DEFAULT_OUTPUT and args.backend == "summary":
        output_path = DEFAULT_OUTPUT_SUMMARY

    figsize = tuple(args.figsize) if args.figsize else DEFAULT_FIGSIZE
    output_path = generate_diagram(
        args.backend,
        model,
        input_hw,
        output_path,
        device,
        figsize=figsize,
        font_scale=args.font_scale,
        grayscale=args.grayscale,
    )
    LOGGER.info("Auto diagram saved to: %s", output_path)


if __name__ == "__main__":
    main()

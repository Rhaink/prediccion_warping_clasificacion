# Thesis Visualization Assets

This guide shows how to generate thesis-ready images for the pipeline and models using:
- NetworkX (graph/pipeline diagrams)
- Graphviz via pydot (flow diagrams)
- Keras/TensorFlow plot_model or ann_visualizer (NN architectures)
- BertViz (transformer attention views)

## Optional installs

Graph diagrams:
```
pip install networkx pydot graphviz
```

Graphviz binaries are also required by pydot/plot_model:
- Debian/Ubuntu: `sudo apt-get install graphviz`
- macOS (brew): `brew install graphviz`
- Windows: install Graphviz and add it to PATH

Keras/ANN Visualizer:
```
pip install tensorflow ann_visualizer
```

BertViz + Transformers:
```
pip install bertviz transformers sentencepiece
```

## NetworkX: pipeline diagram

```python
from src_v2.visualization.diagramming import save_pipeline_networkx_diagram

save_pipeline_networkx_diagram(
    "outputs/thesis_figures/pipeline_networkx.png",
    layout="dot",
    title="Pipeline Overview",
)
```

## Graphviz (pydot): flow diagram

```python
from src_v2.visualization.diagramming import save_pipeline_graphviz_diagram

save_pipeline_graphviz_diagram(
    "outputs/thesis_figures/pipeline_graphviz.svg",
    rankdir="LR",
    title="Pipeline Overview",
)
```

## Keras/TensorFlow plot_model (architecture)

```python
from tensorflow.keras.models import load_model
from src_v2.visualization.diagramming import save_keras_model_diagram

model = load_model("path/to/model.keras")
save_keras_model_diagram(
    model,
    "outputs/thesis_figures/model_architecture.png",
    prefer="plot_model",
)
```

## ANN Visualizer (architecture)

```python
from src_v2.visualization.diagramming import save_keras_model_diagram

save_keras_model_diagram(
    model,
    "outputs/thesis_figures/model_architecture.gv",
    prefer="ann_visualizer",
)
```

ANN Visualizer writes Graphviz files (`.gv`) and may generate a `.gv.pdf`.
Convert to PNG/SVG with Graphviz if needed.

## BertViz: attention visualization

```python
from transformers import AutoTokenizer, AutoModel
from src_v2.visualization.diagramming import save_bertviz_attention_html

model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Texto de ejemplo", return_tensors="pt")
outputs = model(**inputs)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

save_bertviz_attention_html(
    outputs.attentions,
    tokens,
    "outputs/thesis_figures/bert_attention.html",
)
```

If HTML export is not supported by your bertviz version, run `head_view` in a notebook.

## Customizing the pipeline diagram

Edit the nodes and edges in:
`src_v2/visualization/diagramming.py`

Update `PIPELINE_NODES`, `PIPELINE_EDGES`, and the `PIPELINE_COLORS` mapping to
match your thesis narrative (data, warping, training, evaluation, results).

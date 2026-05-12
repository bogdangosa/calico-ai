from torchview import draw_graph
import torch.nn as nn


def visualize_with_torchview(model: nn.Module, input_size: tuple, depth: int = 2):
    """
    Generates a visual graph of the model.
    depth: How many nested layers to show (increase for more detail).
    """
    model_graph = draw_graph(
        model,
        input_size=input_size,
        expand_nested=True,
        depth=depth,
        device='cpu'  # Use 'meta' for huge models to save memory
    )

    # In Jupyter, this will display the graph automatically
    return model_graph.visual_graph

# Example Usage:
# graph = visualize_with_torchview(my_model, (1, 3, 9, 9))
# graph.render("model_diagram", format="png") # Optional: save to file
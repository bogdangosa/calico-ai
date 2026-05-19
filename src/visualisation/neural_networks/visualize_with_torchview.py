from torchview import draw_graph
import torch.nn as nn

from src.machine_learning.networks.dual_head_res_net import DualHeadResNet


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
model = DualHeadResNet(input_channels=3, board_size=7, num_actions=22)
graph = visualize_with_torchview(model, (1, 3, 7, 7))
graph.render("model_diagram", format="png") # Optional: save to file
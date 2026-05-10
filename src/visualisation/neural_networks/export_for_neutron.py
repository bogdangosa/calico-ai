import torch
import torch.nn as nn

from src.machine_learning.networks.dual_head_res_net import DualHeadResNet


def export_for_netron(model: nn.Module, input_size: tuple, filename: str = "model.onnx"):
    """
    Exports any PyTorch model to ONNX format for visualization in Netron.
    input_size: (Batch, Channels, Height, Width)
    """
    model.eval()
    dummy_input = torch.randn(*input_size)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        print(f"✅ Success! Model exported to {filename}. Load it at netron.app")
    except Exception as e:
        print(f"❌ Export failed: {e}")

# Example Usage:
my_model = DualHeadResNet(input_channels=3, board_size=9, num_actions=81)
export_for_netron(my_model, (1, 3, 9, 9))
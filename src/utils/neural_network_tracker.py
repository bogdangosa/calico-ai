import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class NeuralTensorTracker:
    def __init__(self, log_dir: str = "runs/tensor_tracker", 
                 activations_log_freq: int = 100,
                 weights_log_freq: int = 100):
        self.writer = SummaryWriter(log_dir)
        self.hooks = []
        self.global_step = 0
        self.target_types = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
        self.activations_log_freq = activations_log_freq
        self.weights_log_freq = weights_log_freq

    def _register_layer_hook(self, name: str, module: nn.Module):
        def forward_hook(mod, input_tensor, output_tensor):
            # Only log activations every X steps
            if not module.training or self.global_step % self.activations_log_freq != 0:
                return

            # Heavy operations only happen if we are logging
            out_data = output_tensor.detach().cpu()

            self.writer.add_histogram(f"Activations_Distribution/{name}", out_data, self.global_step)
            self.writer.add_scalar(f"Activations_Mean/{name}", out_data.mean().item(), self.global_step)
            self.writer.add_scalar(f"Activations_Std/{name}", out_data.std().item(), self.global_step)

            sparsity = (out_data == 0).float().mean().item()
            self.writer.add_scalar(f"Activations_Sparsity/{name}", sparsity, self.global_step)

            if out_data.dim() == 4:
                grid_img = out_data[0:1, 0:3]
                if grid_img.shape[1] == 1:
                    grid_img = grid_img.repeat(1, 3, 1, 1)

                min_v, max_v = grid_img.min(), grid_img.max()
                if max_v > min_v:
                    grid_img = (grid_img - min_v) / (max_v - min_v)
                    self.writer.add_images(f"Feature_Maps/{name}", grid_img, self.global_step)

        hook_handle = module.register_forward_hook(forward_hook)
        self.hooks.append(hook_handle)

    def attach(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.target_types):
                clean_name = name.replace(".", "/")
                self._register_layer_hook(clean_name, module)
        return self

    def log_weights(self, model: nn.Module, step: int):
        self.global_step = step
        # Only log weights every X steps
        if step % self.weights_log_freq != 0:
            return

        for name, param in model.named_parameters():
            clean_name = name.replace(".", "/")
            param_data = param.detach().cpu()

            if "weight" in name:
                self.writer.add_histogram(f"Weights_Distribution/{clean_name}", param_data, self.global_step)
                self.writer.add_scalar(f"Weights_Norm/{clean_name}", torch.norm(param_data).item(), self.global_step)
            elif "bias" in name:
                self.writer.add_histogram(f"Biases_Distribution/{clean_name}", param_data, self.global_step)

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.writer.close()
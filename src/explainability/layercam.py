import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerCAM:
    """
    Computes LayerCAM visual attribution maps for a given model and target layer.
    Saves positive gradients and activations to map pixel-level diagnostic focus.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks robustly
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        try:
            self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
        except AttributeError:
            self.backward_hook = target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input_t, output_t):
        self.activations = output_t

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        """
        Generates a 2D normalized LayerCAM heatmap.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # If output is evidential (Dirichlet alpha), we sum or use probabilities
        # In either case, we compute gradients with respect to the class score
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        # LayerCAM formulation: pixel-wise weights = max(0, gradients)
        weights = torch.clamp(self.gradients, min=0)
        
        # Weighted activations sum along channels
        cam = weights * self.activations
        cam = torch.sum(cam, dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)  # final ReLU
        
        # Bilinear interpolation back to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Min-max normalize heatmap between [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.squeeze(0).squeeze(0)  # Shape (H, W)

    def release(self):
        """
        Removes hooks to prevent memory accumulation.
        """
        self.forward_hook.remove()
        self.backward_hook.remove()
        
    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

import torch
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, List

class DetectionModel(torch.nn.Module):
    """GPU-optimized lightweight detection model for audio deepfake classification."""

    def __init__(self, config: Config, input_dim: int):
        super(DetectionModel, self).__init__()
        self.config = config
        self.device = config.device

        # GPU optimization settings
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', False)
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        self.use_batch_norm = getattr(config, 'use_batch_norm', True)
        self.use_layer_norm = getattr(config, 'use_layer_norm', False)
        self.fuse_activations = getattr(config, 'fuse_activations', True)

        # Create layer dimensions list
        self.layers_dims = [input_dim] + config.detection_hidden_dims + [1]
        self.num_layers = len(self.layers_dims) - 1

        # Build optimized model architecture
        self._build_model()

        # Initialize parameters for better GPU performance
        self._initialize_parameters()

        # Move to GPU
        self.to(self.device)

        # Compile for inference if available (PyTorch 2.0+)
        if hasattr(torch, 'compile') and getattr(config, 'compile_model', False):
            self.forward = torch.compile(self.forward)

    def _build_model(self):
        """Build GPU-optimized model architecture."""
        layers = []

        for i in range(self.num_layers):
            input_size = self.layers_dims[i]
            output_size = self.layers_dims[i + 1]

            # Linear layer
            linear = nn.Linear(input_size, output_size)
            layers.append(linear)

            # Skip activation and normalization for output layer
            if i < self.num_layers - 1:
                # Normalization layer (choose between batch norm and layer norm)
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(output_size))
                elif self.use_layer_norm:
                    layers.append(nn.LayerNorm(output_size))

                # Activation function
                if self.fuse_activations:
                    # Use inplace ReLU for memory efficiency
                    layers.append(nn.ReLU(inplace=True))
                else:
                    layers.append(nn.ReLU())

                # Dropout
                dropout_rate = getattr(self.config, 'detection_dropout', 0.1)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        self.model = nn.Sequential(*layers)

        # Create separate components for gradient checkpointing
        if self.use_gradient_checkpointing:
            self._create_checkpointed_layers()

    def _create_checkpointed_layers(self):
        """Create layers for gradient checkpointing."""
        self.checkpointed_layers = nn.ModuleList()

        # Group layers into checkpointed blocks
        current_block = []
        for i, layer in enumerate(self.model):
            current_block.append(layer)

            # Create checkpoint every few layers or at the end
            if len(current_block) >= 3 or i == len(self.model) - 1:
                self.checkpointed_layers.append(nn.Sequential(*current_block))
                current_block = []

    def _initialize_parameters(self):
        """Initialize parameters with GPU-friendly values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization works well with ReLU and GPUs
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GPU-optimized forward pass through the detection model."""
        # Ensure tensor is on correct device and dtype
        if x.device != self.device:
            x = x.to(self.device)

        # Mixed precision support
        # if self.use_mixed_precision and x.dtype != torch.float16:
        #     x = x.half()

        # Use gradient checkpointing if enabled and in training mode
        if self.use_gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x)
        else:
            return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.model(x)
        return x.squeeze(-1)

    def _forward_with_checkpointing(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing for memory efficiency."""
        for block in self.checkpointed_layers:
            x = checkpoint(block, x)
        return x.squeeze(-1)

    def forward_batch(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Process multiple inputs in a single batch for better GPU utilization.

        Args:
            x_list: List of input tensors

        Returns:
            Batched output tensor
        """
        # Stack inputs into batch
        batch_x = torch.stack(x_list).to(self.device)
        return self.forward(batch_x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities with GPU optimization.

        Args:
            x: Input tensor

        Returns:
            Probability tensor
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            return probabilities

    def predict_batch_proba(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """Batch probability prediction for better GPU utilization."""
        with torch.no_grad():
            batch_x = torch.stack(x_list).to(self.device)
            logits = self.forward(batch_x)
            probabilities = torch.sigmoid(logits)
            return probabilities

    def get_layer_activations(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Get activations from a specific layer for analysis.

        Args:
            x: Input tensor
            layer_idx: Index of layer to extract activations from

        Returns:
            Layer activations
        """
        with torch.no_grad():
            x = x.to(self.device)

            for i, layer in enumerate(self.model):
                x = layer(x)
                if i == layer_idx:
                    return x

            return x

    def compute_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance using gradients (GPU-optimized).

        Args:
            x: Input tensor

        Returns:
            Feature importance scores
        """
        x = x.to(self.device)
        x.requires_grad_(True)

        output = self.forward(x)
        output.backward()

        # Use gradient magnitude as importance
        importance = torch.abs(x.grad)
        return importance

    def get_model_complexity(self) -> dict:
        """Get model complexity metrics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Calculate FLOPs approximation
        flops = 0
        for i in range(len(self.layers_dims) - 1):
            # Linear layer FLOPs: input_dim * output_dim * 2 (multiply + add)
            flops += self.layers_dims[i] * self.layers_dims[i + 1] * 2

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'approximate_flops': flops,
            'num_layers': self.num_layers,
            'memory_usage_mb': self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate GPU memory usage in MB."""
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())

        # Rough estimate including gradients and activations
        total_memory = param_memory * 3  # params + gradients + some activations
        return total_memory / (1024 * 1024)  # Convert to MB

    def optimize_for_inference(self):
        """Optimize the model for inference."""
        self.eval()

        # Fuse batch norm if present
        if self.use_batch_norm:
            self._fuse_batch_norm()

        # Set to inference mode optimizations
        for module in self.modules():
            if hasattr(module, 'inference_mode'):
                module.inference_mode = True

    def _fuse_batch_norm(self):
        """Fuse batch normalization with linear layers for inference speedup."""
        # This is a simplified version - in practice, you'd need more sophisticated fusion
        for i, module in enumerate(self.model):
            if isinstance(module, nn.BatchNorm1d):
                # Find the preceding linear layer
                if i > 0 and isinstance(self.model[i-1], nn.Linear):
                    linear = self.model[i-1]
                    bn = module

                    # Fuse weights and biases
                    with torch.no_grad():
                        # This is a simplified fusion - real implementation would be more complex
                        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                        linear.weight.mul_(scale.unsqueeze(1))
                        if linear.bias is not None:
                            linear.bias.mul_(scale).add_(bn.bias)
                        else:
                            linear.bias = nn.Parameter(bn.bias.clone())

    def profile_performance(self, input_shape: tuple, num_iterations: int = 100):
        """Profile the model performance for optimization."""
        import time

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)

        # Warm up
        self.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(dummy_input)

        torch.cuda.synchronize()

        # Time the forward pass
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.forward(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations
        throughput = input_shape[0] / avg_time if len(input_shape) > 0 else 1 / avg_time

        complexity = self.get_model_complexity()

        print(f"Performance Profile:")
        print(f"  Average forward pass time: {avg_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
        print(f"  Model parameters: {complexity['total_parameters']:,}")
        print(f"  Estimated FLOPs: {complexity['approximate_flops']:,}")

    def save_optimized(self, filepath: str):
        """Save model in an optimized format."""
        # Save with optimizations
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'layers_dims': self.layers_dims,
                'use_batch_norm': self.use_batch_norm,
                'use_layer_norm': self.use_layer_norm,
                'fuse_activations': self.fuse_activations,
            },
            'complexity': self.get_model_complexity()
        }, filepath)

    @classmethod
    def load_optimized(cls, filepath: str, config: Config):
        """Load model from optimized format."""
        checkpoint = torch.load(filepath, map_location=config.device)

        # Update config with saved settings
        for key, value in checkpoint['config'].items():
            setattr(config, key, value)

        # Create model and load state
        input_dim = checkpoint['config']['layers_dims'][0]
        model = cls(config, input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model


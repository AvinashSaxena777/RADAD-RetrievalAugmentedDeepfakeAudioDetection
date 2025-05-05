import torch
from config import Config
class DetectionModel(torch.nn.Module):
    """Lightweight detection model for audio deepfake classification."""
    
    def __init__(self, config: Config, input_dim: int):
        super(DetectionModel, self).__init__()
        self.config = config
        
        # Create layer dimensions list
        layers_dims = [input_dim] + config.detection_hidden_dims + [1]
        
        # Create sequential model with dropout
        layers = []
        for i in range(len(layers_dims) - 1):
            layers.append(torch.nn.Linear(layers_dims[i], layers_dims[i+1]))
            
            # Add ReLU and dropout for all but the last layer
            if i < len(layers_dims) - 2:
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(config.detection_dropout))
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the detection model."""
        return self.model(x).squeeze(-1)

import torch
from config import Config
class ProjectionLayer(torch.nn.Module):
    """Implements the attention-based projection layer for audio deepfake detection."""
    
    def __init__(self, config: Config, input_dim: int):
        super(ProjectionLayer, self).__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Attention score computation layer
        self.attention_score = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config.projection_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(config.projection_hidden_dim, 1)
        )
        
        # Channel-wise spectral-temporal attention
        self.cst_attention = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config.projection_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.projection_hidden_dim, input_dim)
        )
        
        # Weighted summation layer
        self.weight_sum = torch.nn.Linear(input_dim, config.projection_hidden_dim)
        
        # Normalization layer
        self.normalization = torch.nn.LayerNorm(config.projection_hidden_dim)
        
        # Output layer for unified embedding
        self.unified_embedding = torch.nn.Linear(config.projection_hidden_dim, config.projection_output_dim)
    
    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection layer.
        
        Args:
            input_embeddings: Tensor of shape [batch_size, top_k, embedding_dim]
                              where top_k is the number of retrieved embeddings
        
        Returns:
            Tensor of shape [batch_size, projection_output_dim]
        """
        # Calculate attention scores
        # [batch_size, top_k, 1]
        attention_scores = self.attention_score(input_embeddings)
        
        # Apply softmax to get attention weights
        # [batch_size, top_k, 1]
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        
        # Apply channel-wise spectral-temporal attention
        # [batch_size, top_k, embedding_dim]
        cst_output = self.cst_attention(input_embeddings)
        
        # Apply attention weights to the CST output
        # [batch_size, top_k, embedding_dim]
        weighted_embeddings = cst_output * attention_weights
        
        # Sum across the top_k dimension
        # [batch_size, embedding_dim]
        summed_embeddings = torch.sum(weighted_embeddings, dim=1)
        
        # Apply weighted summation layer
        # [batch_size, projection_hidden_dim]
        weighted_sum = self.weight_sum(summed_embeddings)
        
        # Apply normalization
        # [batch_size, projection_hidden_dim]
        normalized = self.normalization(weighted_sum)
        
        # Create unified embedding
        # [batch_size, projection_output_dim]
        unified = self.unified_embedding(normalized)
        
        return unified

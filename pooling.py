import torch
from config import Config
class TemporalPyramidPooling:
    """Implement Temporal Pyramid Pooling for audio features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.levels = config.tpp_levels
        self.pooling_type = config.tpp_pooling_type
    
    def _pool_segment(self, feature: torch.Tensor, level: int) -> torch.Tensor:
        """Pool a single segment at the given level."""
        # feature shape: [sequence_length, feature_dim]
        seq_len, feat_dim = feature.shape
        
        # Calculate number of bins for this level
        num_bins = level
        
        # Calculate size of each bin (allowing for overlapping bins if needed)
        bin_size = seq_len // num_bins
        
        # Initialize output tensor
        pooled = torch.zeros(num_bins, feat_dim)
        
        # Pool each bin
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = min(start_idx + bin_size, seq_len)
            
            if self.pooling_type == "max":
                pooled[i] = torch.max(feature[start_idx:end_idx], dim=0)[0]
            elif self.pooling_type == "avg":
                pooled[i] = torch.mean(feature[start_idx:end_idx], dim=0)
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        return pooled.flatten()
    
    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply temporal pyramid pooling to features."""
        # features shape: [sequence_length, feature_dim]
        pooled_outputs = []
        
        for level in self.levels:
            pooled = self._pool_segment(features, level)
            pooled_outputs.append(pooled)
        
        # Concatenate all pooled outputs
        return torch.cat(pooled_outputs)
    
    def get_output_dim(self) -> int:
        """Calculate the dimension of the TPP output."""
        # For each level, we have level * feature_dim values
        return sum(self.levels) * self.config.feature_dim

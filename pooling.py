import torch
from config import Config
from typing import List
class TemporalPyramidPooling:
    """Implement Temporal Pyramid Pooling for audio features with GPU optimization."""

    def __init__(self, config: Config):
        self.config = config
        self.levels = config.tpp_levels
        self.pooling_type = config.tpp_pooling_type
        self.device = config.device  # Store device reference

    def _pool_segment_vectorized(self, feature: torch.Tensor, level: int) -> torch.Tensor:
        """Vectorized pooling for a single segment at the given level."""
        # feature shape: [sequence_length, feature_dim]
        seq_len, feat_dim = feature.shape

        # Calculate number of bins for this level
        num_bins = level
        bin_size = seq_len // num_bins

        # Handle edge case where bin_size is 0
        if bin_size == 0:
            # If sequence is shorter than number of bins, duplicate the feature
            if self.pooling_type == "max":
                return feature.max(dim=0)[0].repeat(num_bins).to(self.device)
            else:  # avg
                return feature.mean(dim=0).repeat(num_bins).to(self.device)

        # Create indices for vectorized pooling - keep on GPU
        indices = torch.arange(seq_len, device=self.device)
        bin_indices = indices // bin_size
        # Clip bin indices to handle remainder
        bin_indices = torch.clamp(bin_indices, max=num_bins - 1)

        # Initialize output tensor on correct device
        pooled = torch.zeros(num_bins, feat_dim, device=self.device, dtype=feature.dtype)

        # Vectorized pooling using scatter operations
        if self.pooling_type == "max":
            # Use scatter_reduce for max pooling (PyTorch 1.12+)
            if hasattr(torch, 'scatter_reduce'):
                pooled = pooled.scatter_reduce(0, bin_indices.unsqueeze(1).expand(-1, feat_dim),
                                             feature, reduce='amax', include_self=False)
            else:
                # Fallback for older PyTorch versions
                for i in range(num_bins):
                    mask = (bin_indices == i)
                    if mask.any():
                        pooled[i] = feature[mask].max(dim=0)[0]

        elif self.pooling_type == "avg":
            # Use scatter_add for average pooling
            pooled = pooled.scatter_add(0, bin_indices.unsqueeze(1).expand(-1, feat_dim), feature)

            # Count elements in each bin for averaging
            bin_counts = torch.bincount(bin_indices, minlength=num_bins).float()
            bin_counts = bin_counts.clamp(min=1)  # Avoid division by zero
            pooled = pooled / bin_counts.unsqueeze(1)

        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        return pooled.flatten()

    def _pool_segment_efficient(self, feature: torch.Tensor, level: int) -> torch.Tensor:
        """Efficient pooling using adaptive pooling operations."""
        # feature shape: [sequence_length, feature_dim]
        seq_len, feat_dim = feature.shape

        # Reshape for adaptive pooling: [1, feat_dim, seq_len]
        feature_reshaped = feature.t().unsqueeze(0)  # [1, feat_dim, seq_len]

        if self.pooling_type == "max":
            # Use adaptive max pooling - more efficient on GPU
            pooled = torch.nn.functional.adaptive_max_pool1d(feature_reshaped, level)
        elif self.pooling_type == "avg":
            # Use adaptive average pooling - more efficient on GPU
            pooled = torch.nn.functional.adaptive_avg_pool1d(feature_reshaped, level)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        # Reshape back and flatten: [1, feat_dim, level] -> [level, feat_dim] -> [level * feat_dim]
        pooled = pooled.squeeze(0).t().flatten()  # [level * feat_dim]

        return pooled

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply temporal pyramid pooling to features with GPU optimization."""
        # Ensure features are on correct device
        if features.device != self.device:
            features = features.to(self.device)

        # features shape: [sequence_length, feature_dim]
        pooled_outputs = []

        # Use efficient adaptive pooling
        for level in self.levels:
            pooled = self._pool_segment_efficient(features, level)
            pooled_outputs.append(pooled)

        # Concatenate all pooled outputs (already on GPU)
        return torch.cat(pooled_outputs)

    def pool_features_batch(self, features_batch: List[torch.Tensor]) -> torch.Tensor:
        """Process multiple feature tensors in batch for better GPU utilization."""
        if not features_batch:
            return torch.empty(0, device=self.device)

        # Process all features and stack results
        batch_results = []
        for features in features_batch:
            pooled = self.pool_features(features)
            batch_results.append(pooled)

        # Stack into batch tensor [batch_size, total_pooled_dim]
        return torch.stack(batch_results)

    def get_output_dim(self) -> int:
        """Calculate the dimension of the TPP output."""
        # For each level, we have level * feature_dim values
        return sum(self.levels) * self.config.feature_dim


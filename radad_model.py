from config import Config
import torch
from projection import ProjectionLayer
import torch.nn as nn
from detection_model import DetectionModel
# ===========================
# RADAD model (single module)
# ===========================
class RADADModel(nn.Module):
    """
    Retrieval-Augmented Deepfake Audio Detector (single module).
    Encapsulates:
      - ProjectionLayer: processes [B, K, D_enc] retrieved neighbors -> [B, D_proj]
      - fuse: Linear([D_tpp + D_proj] -> D_proj)
      - DetectionModel: MLP head on [B, D_proj] -> logits [B] (or [B,1])
    """
    def __init__(self, config: 'Config', tpp_output_dim: int):
        super().__init__()
        self.config = config
        self.device = config.device

        # submodules
        self.projection_layer = ProjectionLayer(config, tpp_output_dim)
        fuse_in_dim  = tpp_output_dim + config.projection_output_dim
        fuse_out_dim = config.projection_output_dim
        self.fuse = nn.Linear(fuse_in_dim, fuse_out_dim)
        self.detection_model = DetectionModel(config, fuse_out_dim)

        # move to device
        self.to(self.device)

    def forward(self, neighbor_vecs: torch.Tensor, tpp_vecs: torch.Tensor) -> torch.Tensor:
        """
        neighbor_vecs: [B, K, D_enc]
        tpp_vecs:      [B, D_enc]  (pooled query)
        returns: logits [B] or [B,1] (same as DetectionModel)
        """
        proj = self.projection_layer(neighbor_vecs)       # [B, D_proj]
        fused = self.fuse(torch.cat([tpp_vecs, proj], dim=1))  # [B, D_proj]
        logits = self.detection_model(fused)              # [B] or [B,1]
        return logits
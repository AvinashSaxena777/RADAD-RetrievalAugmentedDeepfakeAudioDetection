import torch
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ProjectionLayer(nn.Module):
    """GPU-optimized attention-based projection layer for audio deepfake detection."""

    def __init__(self, config, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.device = getattr(config, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Flags
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', False)
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        self.fuse_operations = getattr(config, 'fuse_attention_ops', True)
        self.norm_in_fp32 = True  # improves stability under AMP

        # Dims
        self.hidden_dim = config.projection_hidden_dim
        self.output_dim = config.projection_output_dim

        # Attention score path
        if self.fuse_operations:
            self.attention_score = nn.Linear(input_dim, self.hidden_dim)
            self.attention_final = nn.Linear(self.hidden_dim, 1)
        else:
            self.attention_score = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, 1),
            )

        # Channel-wise spectral-temporal (CST) path
        if self.fuse_operations:
            self.cst_hidden = nn.Linear(input_dim, self.hidden_dim)
            self.cst_output = nn.Linear(self.hidden_dim, input_dim)
        else:
            self.cst_attention = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, input_dim),
            )

        # Projection head
        self.weight_sum = nn.Linear(input_dim, self.hidden_dim)
        self.normalization = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.unified_embedding = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(getattr(config, 'projection_dropout', 0.1))

        self._initialize_parameters()
        self.to(self.device)

    def _initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _fused_attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attention_score(x)
        h = torch.tanh(h)
        return self.attention_final(h)  # [B, K, 1]

    def _fused_cst_attention(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cst_hidden(x)
        h = F.relu(h, inplace=True)
        return self.cst_output(h)       # [B, K, D]

    def _efficient_attention(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        # input_embeddings: [B, K, D]
        if self.fuse_operations:
            attn_scores = self._fused_attention_scores(input_embeddings)  # [B, K, 1]
            cst_out     = self._fused_cst_attention(input_embeddings)    # [B, K, D]
        else:
            attn_scores = self.attention_score(input_embeddings)         # [B, K, 1]
            cst_out     = self.cst_attention(input_embeddings)           # [B, K, D]

        attn_weights = F.softmax(attn_scores, dim=1)                     # [B, K, 1]
        weighted = attn_weights * cst_out                                # broadcast -> [B, K, D]
        summed = weighted.sum(dim=1)                                     # [B, D]
        return summed

    def _process_embeddings(self, summed_embeddings: torch.Tensor) -> torch.Tensor:
        # Linear -> (optional fp32) LayerNorm -> Dropout -> Linear
        x = self.weight_sum(summed_embeddings)                            # [B, H]
        if self.norm_in_fp32 and x.dtype != torch.float32:
            x_fp32 = x.float()
            x_norm = self.normalization(x_fp32).to(x.dtype)
        else:
            x_norm = self.normalization(x)
        x_norm = self.dropout(x_norm)
        out = self.unified_embedding(x_norm)                              # [B, output_dim]
        return out

    def _forward_impl(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        summed = self._efficient_attention(input_embeddings)
        return self._process_embeddings(summed)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        # Expect [B, top_k, D]
        if input_embeddings.device != self.device:
            input_embeddings = input_embeddings.to(self.device)

        # Let caller control autocast; do NOT manually .half() here
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, input_embeddings)
        else:
            return self._forward_impl(input_embeddings)

    def forward_batch(self, input_embeddings_list: list) -> torch.Tensor:
        # Each: [K, D] -> stack to [B, K, D]
        batch_embeddings = torch.stack(input_embeddings_list).to(self.device)
        return self.forward(batch_embeddings)

    @torch.no_grad()
    def get_attention_weights(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        if self.fuse_operations:
            scores = self._fused_attention_scores(input_embeddings)
        else:
            scores = self.attention_score(input_embeddings)
        return F.softmax(scores, dim=1)  # [B, K, 1]

    def memory_efficient_forward(self, input_embeddings: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
        if input_embeddings.size(0) <= chunk_size:
            return self.forward(input_embeddings)
        outs = []
        for i in range(0, input_embeddings.size(0), chunk_size):
            outs.append(self.forward(input_embeddings[i:i+chunk_size]))
        return torch.cat(outs, dim=0)

    def profile_performance(self, input_shape: tuple, num_iterations: int = 100):
        import time
        dummy = torch.randn(input_shape, device=self.device)
        for _ in range(10):
            _ = self.forward(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(num_iterations):
            _ = self.forward(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = (t0 - time.time()) / max(1, num_iterations)
        print(f"Avg forward: {abs(dt)*1000:.2f} ms/iter")

    def get_flops(self, input_shape: tuple) -> int:
        B, K, D = input_shape
        flops = B * K * (D * self.hidden_dim + self.hidden_dim)          # attention_score + attention_final
        flops += B * K * (D * self.hidden_dim + self.hidden_dim * D)     # CST path
        flops += B * (D * self.hidden_dim + self.hidden_dim * self.output_dim)  # head
        return flops

import torch
import os
class Config:
    """Configuration class for the deepfake detection system."""

    def __init__(self):
        # Data paths
        self.data_root = "./data"
        self.train_data_path = "/release_in_the_wild"
        self.test_data_path = "/release_in_the_wild"  # Same source but will be split
        self.vector_db_path = os.path.join(self.data_root, "vector_db")

        # Data loading and splitting
        self.data_fraction = 1.0
        self.train_split = 0.8  # 80% for training, 20% for validation
        self.random_seed = 42

        # Add a flag to prevent data leakage
        self.prevent_data_leakage = True

        # Audio processing
        self.sample_rate = 16000  # Target sample rate
        self.segment_length = 2.0  # Segment length in seconds
        self.segment_overlap = 0.5  # Overlap between segments (0-1)

        # Wav2Vec2 model
        self.wav2vec2_model_name = "facebook/wav2vec2-base-960h"
        self.wav2vec2_layers_to_use = [-4, -3, -2, -1]  # Which layers to use for feature extraction

        # Temporal Pyramid Pooling
        self.tpp_levels = [1, 2, 4]  # Pyramid levels
        self.tpp_pooling_type = "max"  # 'max' or 'avg'

        # Vector database
        self.vector_db_index_type = "L2"  # L2 distance for similarity search
        self.vector_db_nprobe = 10  # Number of cells to visit during search

        # Retrieval
        self.top_k = 5  # Number of similar vectors to retrieve

        # Projection layer
        self.projection_hidden_dim = 256
        self.projection_output_dim = 128

        # Detection model
        self.detection_hidden_dims = [64, 32]
        self.detection_dropout = 0.2

        # Training
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 5
        self.early_stopping_patience = 5

        self.faiss_gpu_memory_fraction: float = 0.8  # Use 80% of GPU memory
        self.use_float16: bool = False  # Memory optimization
        self.vector_add_batch_size: int = 10000  # Batch size for adding vectors
        self.vector_db_nprobe: int = 32
        self.use_mixed_precision: bool = False  # Enable for memory savings
        self.use_gradient_checkpointing: bool = False  # Enable for large models
        self.fuse_attention_ops: bool = True  # Fuse operations for speed
        self.projection_dropout: float = 0.1  # Regularization

        self.use_batch_norm: bool = True  # Better for small batches
        self.use_layer_norm: bool = False  # Alternative to batch norm
        self.fuse_activations: bool = True  # Memory efficiency
        self.compile_model: bool = False  # PyTorch 2.0+ optimization
        self.detection_dropout: float = 0.1  # Regularization

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SPOOF_VALUES = {'spoof', 'fake', 'synthetic', 'spoofed', 'tts', 'vc', 'voice-conversion', 'voice conversion'}
        self.BONA_VALUES  = {'bona-fide', 'bonafide', 'genuine', 'real', 'authentic', 'bona fide'}

    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
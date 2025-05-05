import torch
import os
class Config:
    """Configuration class for the deepfake detection system."""
    
    def __init__(self):
        # Data paths
        self.data_root = "./data"
        self.train_data_path = os.path.join(self.data_root, "train")
        self.test_data_path = os.path.join(self.data_root, "test")
        self.vector_db_path = os.path.join(self.data_root, "vector_db")
        
        # Data loading
        self.data_fraction = 1.0  # Can be changed to load partial data (e.g., 0.25 for 25%)
        self.random_seed = 42
        
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
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 50
        self.early_stopping_patience = 5
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")
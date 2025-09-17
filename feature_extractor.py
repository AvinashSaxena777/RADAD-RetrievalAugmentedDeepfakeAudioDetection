import torch
import numpy as np
from typing import List
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from config import Config
class Wav2Vec2FeatureExtractor:
    """Extract features from audio using Wav2Vec2 model."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(config.wav2vec2_model_name)
        self.model = Wav2Vec2Model.from_pretrained(config.wav2vec2_model_name).to(self.device)
        self.model.eval()

        # Get feature dimensions
        self.feature_dim = self.model.config.hidden_size

    def extract_features(self, audio_segments: List[np.ndarray], move_to_cpu=False) -> List[torch.Tensor]:
        """Efficiently extract features from audio segments in a batch using Wav2Vec2."""

        # Batch encode segments (padding them to equal length)
        inputs = self.processor(
            audio_segments,  # List of segments
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=True  # Ensures all are same length for batching
        ).input_values.to(self.device)  # Move inputs to GPU

        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if self.config.wav2vec2_layers_to_use:
                selected_layers = [hidden_states[i] for i in self.config.wav2vec2_layers_to_use]
                # Stack and average the selected layers along layer axis
                features = torch.mean(torch.stack(selected_layers), dim=0)
            else:
                features = hidden_states[-1]  # Use last hidden state

            # features shape: (batch_size, seq_len, feature_dim)
            # No need to .cpu() yet if downstream ops are torch/GPU

            # Remove batch dimension for each segment, optionally move to CPU if required
            if move_to_cpu:
                features_list = [feat.cpu() for feat in features]
            else:
                features_list = [feat for feat in features]

        return features_list


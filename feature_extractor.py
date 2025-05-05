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
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get feature dimensions
        self.feature_dim = self.model.config.hidden_size
    
    def extract_features(self, audio_segments: List[np.ndarray]) -> torch.Tensor:
        """Extract features from audio segments using Wav2Vec2."""
        features_list = []
        
        with torch.no_grad():
            for segment in audio_segments:
                # Preprocess audio
                inputs = self.processor(
                    segment, 
                    sampling_rate=self.config.sample_rate, 
                    return_tensors="pt"
                ).input_values.to(self.device)
                
                # Get model outputs
                outputs = self.model(inputs, output_hidden_states=True)
                
                # Extract hidden states from specified layers
                hidden_states = outputs.hidden_states
                
                # Use specific layers if specified, otherwise use the last layer
                if self.config.wav2vec2_layers_to_use:
                    selected_layers = [hidden_states[i] for i in self.config.wav2vec2_layers_to_use]
                    # Average across the selected layers
                    features = torch.mean(torch.stack(selected_layers), dim=0)
                else:
                    # Use the last hidden state
                    features = hidden_states[-1]
                
                # Remove batch dimension and move to CPU
                features = features.squeeze(0).cpu()
                
                features_list.append(features)
        
        return features_list

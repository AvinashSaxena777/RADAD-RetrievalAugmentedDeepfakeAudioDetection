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

class WhisperFeatureExtractor:
    """
    Whisper encoder-based feature extractor.

    - Uses HF WhisperFeatureExtractor to build log-mel inputs
    - Runs WhisperModel encoder; returns last_hidden_state per segment
    - Exposes .feature_dim = model.config.d_model
    """
    def __init__(self, config):
        from transformers import WhisperModel, WhisperFeatureExtractor as HFWhisperFeatureExtractor

        self.config = config
        self.device = config.device
        self.model_name = getattr(config, "whisper_model_name", "openai/whisper-small")

        # HF preprocess + model
        self.processor = HFWhisperFeatureExtractor.from_pretrained(self.model_name)
        self.model = WhisperModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # we don't fine-tune Whisper here

        # d_model is the frame embedding dim the encoder outputs
        self.feature_dim = int(getattr(self.model.config, "d_model", 768))

        # For Whisper, ensure 16k sample rate is used by your loader
        if getattr(self.config, "sample_rate", 16000) != 16000:
            import logging
            logging.warning("Whisper expects 16 kHz; consider setting config.sample_rate = 16000.")

        # Use mixed precision only on CUDA
        self._use_amp = bool(getattr(self.config, "use_mixed_precision", False)) and self.device.type == "cuda"

    def extract_features(self, segments: "List[np.ndarray]") -> "List[torch.Tensor]":
        """
        segments: list of 1D float numpy arrays at 16kHz.
        returns: list of tensors [T_i, feature_dim] on CPU (pipeline will .to(device))
        """
        feats = []
        for wav in segments:
            # HF FE expects float array at 16 kHz
            inputs = self.processor(
                wav, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)  # [1, 80, n_frames]

            # Run encoder only
            if self._use_amp:
                from torch.amp import autocast
                amp_ctx = autocast("cuda")
            else:
                class _Null:
                    def __enter__(self): return None
                    def __exit__(self, *a): return False
                amp_ctx = _Null()

            with torch.no_grad(), amp_ctx:
                enc_out = self.model.encoder(input_features)  # BaseModelOutput
                hs = enc_out.last_hidden_state  # [1, T, d_model]
                hs = hs.squeeze(0).detach().cpu()  # [T, d_model] on CPU to match your pipeline
                feats.append(hs)

        return feats

class WavLMFeatureExtractor:
    """
    WavLM encoder-based feature extractor using HF Transformers.
    - Input: list of 1D float arrays @ 16 kHz
    - Output: list of Tensors [T_i, hidden_size] on CPU (your pipeline moves them to device)
    """
    def __init__(self, config):
        from transformers import WavLMModel, AutoFeatureExtractor

        self.config = config
        self.device = config.device
        self.model_name = getattr(config, "wavlm_model_name", "microsoft/wavlm-base-plus")

        # HF frontend + model
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = WavLMModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # hidden size of encoder outputs
        self.feature_dim = int(getattr(self.model.config, "hidden_size", 768))

        # AMP only if CUDA + enabled
        self._use_amp = bool(getattr(self.config, "use_mixed_precision", False)) and self.device.type == "cuda"

        # WavLM is trained for 16 kHz
        if getattr(self.config, "sample_rate", 16000) != 16000:
            logging.warning("WavLM expects 16 kHz; set config.sample_rate = 16000 for best results.")

    def extract_features(self, segments: List["np.ndarray"]) -> List[torch.Tensor]:
        feats = []
        for wav in segments:
            # Build inputs
            inputs = self.processor(raw_speech=wav, sampling_rate=16000, return_tensors="pt")
            input_values   = inputs.input_values.to(self.device)     # [1, T]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # AMP context if requested
            if self._use_amp:
                from torch.amp import autocast
                ctx = autocast("cuda")
            else:
                class _Null:
                    def __enter__(self): return None
                    def __exit__(self, *a): return False
                ctx = _Null()

            with torch.no_grad(), ctx:
                out = self.model(input_values, attention_mask=attention_mask)
                hs = out.last_hidden_state.squeeze(0).detach().cpu()  # [T, hidden]
            feats.append(hs)
        return feats
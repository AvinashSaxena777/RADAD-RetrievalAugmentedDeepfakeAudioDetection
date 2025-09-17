import numpy as np
from typing import List, Dict, Any
from config import Config
class AudioSegmenter:
    """Class for segmenting audio into fixed-length chunks with overlap."""

    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.segment_length = int(config.segment_length * config.sample_rate)
        self.segment_overlap = config.segment_overlap
        self.hop_length = int(self.segment_length * (1 - self.segment_overlap))

    def segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """Segment audio into fixed-length chunks with overlap."""
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            raise ValueError("Expected 1D audio array")

        # Get total number of samples
        total_samples = len(audio)

        # Calculate number of segments
        num_segments = max(1, (total_samples - self.segment_length) // self.hop_length + 1)

        segments = []
        for i in range(num_segments):
            start = i * self.hop_length
            end = min(start + self.segment_length, total_samples)

            # Handle the last segment if it's shorter than segment_length
            segment = audio[start:end]
            if len(segment) < self.segment_length:
                # Pad with zeros if the segment is shorter than segment_length
                padding = np.zeros(self.segment_length - len(segment))
                segment = np.concatenate([segment, padding])

            segments.append(segment)

        # If the audio is shorter than segment_length, just use the padded version
        if not segments:
            segment = audio
            if len(segment) < self.segment_length:
                padding = np.zeros(self.segment_length - len(segment))
                segment = np.concatenate([segment, padding])
            segments.append(segment)

        return segments

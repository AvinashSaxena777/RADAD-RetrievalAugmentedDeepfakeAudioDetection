import os
import random
import logging
import numpy as np
import pandas as pd
import librosa
import torchaudio
from config import Config

class AudioDataset:
    """Dataset class for loading and managing audio data with support for partial loading."""
    
    def __init__(self, config: Config, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        self.data_path = config.train_data_path if is_train else config.test_data_path
        self.sample_rate = config.sample_rate
        
        # Initialize data
        self.audio_files = []
        self.labels = []
        self.metadata = {}
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load audio file paths and labels with support for loading partial dataset."""
        # First try standard metadata file
        metadata_file = os.path.join(self.data_path, "metadata.csv")
        
        # Then try meta.csv (your format)
        if not os.path.exists(metadata_file):
            metadata_file = os.path.join(self.data_path, "meta.csv")
        
        if os.path.exists(metadata_file):
            logging.info(f"Loading metadata from {metadata_file}")
            df = pd.read_csv(metadata_file)
            
            # Handle different column names
            file_col = 'file' if 'file' in df.columns else 'path'
            label_col = 'label' if 'label' in df.columns else 'label'
            
            # Apply data_fraction to limit the amount of data loaded
            if self.config.data_fraction < 1.0:
                # Set seed for reproducibility
                random.seed(self.config.random_seed)
                
                # Stratified sampling to maintain class balance
                # Handle both 'bonafide' and 'bona-fide' formats
                bonafide_values = ['bonafide', 'bona-fide']
                bonafide_samples = df[df[label_col].isin(bonafide_values)]
                spoof_samples = df[~df[label_col].isin(bonafide_values)]
                
                bonafide_count = int(len(bonafide_samples) * self.config.data_fraction)
                spoof_count = int(len(spoof_samples) * self.config.data_fraction)
                
                # Ensure we sample at least 1 item if available
                bonafide_count = max(1, min(bonafide_count, len(bonafide_samples)))
                spoof_count = max(1, min(spoof_count, len(spoof_samples)))
                
                # Only proceed if we have data to sample
                if bonafide_count > 0 and spoof_count > 0:
                    sampled_bonafide = bonafide_samples.sample(bonafide_count)
                    sampled_spoof = spoof_samples.sample(spoof_count)
                    
                    df = pd.concat([sampled_bonafide, sampled_spoof]).reset_index(drop=True)
                    
                    logging.info(f"Loaded {len(df)} samples ({self.config.data_fraction * 100}% of data)")
                    logging.info(f"Bonafide: {len(sampled_bonafide)}, Spoof: {len(sampled_spoof)}")
            
            # Construct full paths to audio files
            self.audio_files = []
            for file_path in df[file_col]:
                if os.path.isabs(file_path):
                    self.audio_files.append(file_path)
                else:
                    self.audio_files.append(os.path.join(self.data_path, file_path))
            
            # Convert labels to numeric (1 for bonafide, 0 for spoof)
            self.labels = []
            for label in df[label_col]:
                if label.lower() in ['bonafide', 'bona-fide']:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
            
            # Store additional metadata if available
            if 'speaker' in df.columns:
                self.metadata['speaker_id'] = df['speaker'].tolist()
        else:
            # If metadata file doesn't exist, scan directory structure
            logging.warning(f"Metadata file not found at {metadata_file}. Scanning directory structure.")
            
            # Initialize empty lists to avoid errors later
            self.audio_files = []
            self.labels = []
            
            # Look for audio files directly in the data path (flat structure)
            audio_files = [f for f in os.listdir(self.data_path) 
                        if f.endswith(('.wav', '.flac'))]
            
            if audio_files:
                logging.warning(f"Found {len(audio_files)} audio files but no labels. Cannot proceed with training.")
                logging.warning("Please provide a metadata.csv or meta.csv file with file paths and labels.")
            else:
                # Try the original directory scanning logic as a last resort
                bonafide_dir = os.path.join(self.data_path, "bonafide")
                spoof_dir = os.path.join(self.data_path, "spoof")
                
                # Load bonafide samples
                if os.path.exists(bonafide_dir):
                    bonafide_files = [os.path.join(bonafide_dir, f) for f in os.listdir(bonafide_dir) 
                                    if f.endswith(('.wav', '.flac'))]
                    self.audio_files.extend(bonafide_files)
                    self.labels.extend([1] * len(bonafide_files))
                
                # Load spoof samples
                if os.path.exists(spoof_dir):
                    spoof_files = [os.path.join(spoof_dir, f) for f in os.listdir(spoof_dir) 
                                if f.endswith(('.wav', '.flac'))]
                    self.audio_files.extend(spoof_files)
                    self.labels.extend([0] * len(spoof_files))
                
                # Apply data_fraction
                if self.audio_files and self.config.data_fraction < 1.0:
                    data_size = len(self.audio_files)
                    subset_size = int(data_size * self.config.data_fraction)
                    
                    # Set seed for reproducibility
                    random.seed(self.config.random_seed)
                    
                    # Create index pairs of (file_path, label)
                    paired_data = list(zip(self.audio_files, self.labels))
                    
                    # Shuffle and select subset
                    random.shuffle(paired_data)
                    paired_data = paired_data[:subset_size]
                    
                    # Critical fix: Check if paired_data is not empty before unpacking
                    if paired_data:
                        # Unzip the pairs
                        self.audio_files, self.labels = zip(*paired_data)
                        logging.info(f"Loaded {len(self.audio_files)} samples ({self.config.data_fraction * 100}% of data)")
                        
                        # Convert tuples to lists for consistency
                        self.audio_files = list(self.audio_files)
                        self.labels = list(self.labels)
        
        # Final check to ensure we have data
        if not self.audio_files:
            logging.error("No audio files found. Please check your dataset structure and metadata file.")
            # Return empty lists rather than raising an error
            self.audio_files = []
            self.labels = []

    
    def __len__(self):
        """Return the number of audio files."""
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """Get audio file path and label."""
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Convert label to numeric
        label_num = 1 if label == 'bonafide' else 0
        
        # Get metadata if available
        metadata = {}
        for key in self.metadata:
            metadata[key] = self.metadata[key][idx]
        
        return {
            'path': audio_path,
            'label': label_num,
            'metadata': metadata
        }
    
    def load_audio(self, audio_path):
        """Load and resample audio file."""
        try:
            # Use librosa to load the audio file
            audio_data, _ = librosa.load(audio_path, sr=self.sample_rate, duration=3.0, mono=True)
            
            # Pad if audio is shorter than the expected duration
            expected_length = int(3.0 * self.sample_rate)  # 3 seconds duration
            if len(audio_data) < expected_length:
                audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
            
            return audio_data
        except Exception as e:
            logging.error(f"Error loading {audio_path}: {e}")
            return None
    # def load_audio(self, audio_path):
    #     """Load and resample audio file."""
    #     # Convert to absolute path
    #     audio_path = os.path.abspath(audio_path)
    #     waveform, sample_rate = torchaudio.load(audio_path)

    #     # Convert to mono if stereo
    #     if waveform.shape[0] > 1:
    #         waveform = torch.mean(waveform, dim=0, keepdim=True)

    #     # Resample if needed
    #     if sample_rate != self.sample_rate:
    #         resampler = torchaudio.transforms.Resample(
    #             orig_freq=sample_rate,
    #             new_freq=self.sample_rate
    #         )
    #         waveform = resampler(waveform)

    #     return waveform.squeeze(0).numpy()
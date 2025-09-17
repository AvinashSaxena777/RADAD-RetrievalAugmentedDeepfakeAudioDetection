import os
import random
import logging
import numpy as np
import pandas as pd
import librosa
import torchaudio
from config import Config
import torch

class AudioDataset:
    """Dataset class with proper train/validation splitting to prevent data leakage."""
    SPOOF_VALUES = {'spoof', 'fake', 'synthetic', 'spoofed', 'tts', 'vc', 'voice-conversion', 'voice conversion'}
    BONA_VALUES  = {'bona-fide', 'bonafide', 'genuine', 'real', 'authentic', 'bona fide'}

    def __init__(self, config: Config, is_train: bool = True, split_data: bool = True):
        self.config = config
        self.is_train = is_train
        self.split_data = split_data
        self.data_path = config.train_data_path
        self.sample_rate = config.sample_rate

        # Initialize data
        self.audio_files = []
        self.labels = []
        self.metadata = {}

        # Load data with proper splitting
        self._load_data_with_split()
        

    def _normalize_label_to_int(self, s: str) -> int:
        s = str(s).strip().lower()
        if s in self.SPOOF_VALUES:
            return 1  # SPOOF = 1 (positive class)
        if s in self.BONA_VALUES:
            return 0  # BONAFIDE = 0
        # If anything unexpected appears, fail fast so you notice
        raise ValueError(f"Unknown label string: {s!r}")

    def _load_data_with_split(self):
      metadata_file = os.path.join(self.data_path, "meta.csv")
      if not os.path.exists(metadata_file):
          raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

      logging.info(f"Loading metadata from {metadata_file}")
      df = pd.read_csv(metadata_file)

      file_col  = 'file'  if 'file'  in df.columns else 'path'
      label_col = 'label' if 'label' in df.columns else 'label'

      # --- Normalize labels once; SPOOF=1, BONAFIDE=0 ---
      df['label_norm'] = df[label_col].astype(str).str.strip().str.lower()
      df['y'] = df['label_norm'].apply(self._normalize_label_to_int)

      # Optional: verify there are only two classes
      unique_y = df['y'].unique().tolist()
      assert set(unique_y) <= {0, 1}, f"Unexpected numeric labels: {unique_y}"

      # Apply class-balanced data fraction if requested
      if getattr(self.config, 'data_fraction', 1.0) < 1.0:
          frac = float(self.config.data_fraction)
          np.random.seed(self.config.random_seed)
          df = (
              df.groupby('y', group_keys=False)
                .apply(lambda g: g.sample(max(1, int(round(len(g)*frac))), random_state=self.config.random_seed))
                .reset_index(drop=True)
          )
          logging.info(f"Applied data fraction {frac*100:.1f}% → {len(df)} samples")

      # Train/validation split
      if self.split_data:
          from sklearn.model_selection import train_test_split
          X = df[[file_col] + [c for c in df.columns if c not in [file_col, label_col, 'y']]]
          y = df['y']
          X_train, X_val, y_train, y_val = train_test_split(
              X, y,
              train_size=self.config.train_split,
              test_size=1.0 - self.config.train_split,
              random_state=self.config.random_seed,
              stratify=y,   # stratify on numeric y (0/1)
          )
          if self.is_train:
              df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
              logging.info(f"Loading TRAINING set: {len(df)} samples")
          else:
              df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
              logging.info(f"Loading VALIDATION set: {len(df)} samples")

      # Use the numeric y when building arrays
      self._process_dataframe(df, file_col, label_col='y')

    def _process_dataframe(self, df, file_col, label_col):
      self._file_ids = df[file_col].tolist()

      # Full paths
      self.audio_files = []
      for file_path in df[file_col]:
          self.audio_files.append(file_path if os.path.isabs(file_path)
                                  else os.path.join(self.data_path, file_path))

      # Numeric labels already prepared: 1=spoof, 0=bonafide
      ys = df[label_col].astype(int).tolist()
      self.labels = ys

      # Metadata
      if 'speaker' in df.columns:
          self.metadata['speaker_id'] = df['speaker'].tolist()

      # Log counts with correct semantics
      n_spoof = sum(1 for v in self.labels if v == 1)
      n_bona  = sum(1 for v in self.labels if v == 0)
      split_type = "TRAINING" if self.is_train else "VALIDATION"
      logging.info(f"{split_type} SET - Total: {len(self.labels)}, Spoof(1): {n_spoof}, Bonafide(0): {n_bona}")


    # Rest of your existing methods remain the same...
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
      audio_path = self.audio_files[idx]
      label = float(self.labels[idx])  # 0.0 or 1.0

      metadata = {k: self.metadata[k][idx] for k in self.metadata}

      return {
          'path': audio_path,
          'label': torch.tensor(label, dtype=torch.float32),  # <- float tensor
          'metadata': metadata
      }


    def load_audio(self, audio_path):
        """Load and resample audio file."""
        try:
            # Use librosa to load the audio file
            audio_data, _ = librosa.load(audio_path, sr=self.sample_rate, duration=3.0, mono=True)

            # Pad if audio is shorter than expected duration
            expected_length = int(3.0 * self.sample_rate)
            if len(audio_data) < expected_length:
                audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))

            return audio_data
        except Exception as e:
            logging.error(f"Error loading {audio_path}: {e}")
            return np.zeros(int(3.0 * self.sample_rate))
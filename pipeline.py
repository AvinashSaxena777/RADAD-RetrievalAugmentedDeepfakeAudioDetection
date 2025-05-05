import os
import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from config import Config
from dataset import AudioDataset
from segmenter import AudioSegmenter
from feature_extractor import Wav2Vec2FeatureExtractor
from vector_database import VectorDatabase
from pooling import TemporalPyramidPooling
from projection import ProjectionLayer
from detection_model import DetectionModel
# from torch.utils.data import DataLoader

class DeepfakeDetectionPipeline:
    """Main pipeline for audio deepfake detection."""

    def __init__(self, config: Config):
        self.config = config

        # Initialize components
        self.audio_segmenter = AudioSegmenter(config)

        # Initialize feature extractor first
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)

        # Set feature_dim in config based on actual loaded model
        if not hasattr(config, 'feature_dim'):
            config.feature_dim = self.feature_extractor.feature_dim
            print(f"Feature dimension set to: {config.feature_dim}")

        # Now initialize TPP with correct feature dimension
        self.tpp = TemporalPyramidPooling(config)
        self.vector_db = VectorDatabase(config)

        # Calculate output dimension of TPP
        self.tpp_output_dim = self.tpp.get_output_dim()

        # Initialize model components
        self.projection_layer = ProjectionLayer(config, self.tpp_output_dim)
        self.detection_model = DetectionModel(config, config.projection_output_dim)

        # Move models to device
        self.projection_layer.to(config.device)
        self.detection_model.to(config.device)

        # Set up optimizers
        self.projection_optimizer = torch.optim.Adam(
            self.projection_layer.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.detection_optimizer = torch.optim.Adam(
            self.detection_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Set up loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def process_audio(self, audio_path: str) -> np.ndarray:
        """Process a single audio file through the pipeline."""
        # Load audio file
        dataset = AudioDataset(self.config, is_train=False)
        audio = dataset.load_audio(audio_path)

        # Segment audio
        audio_segments = self.audio_segmenter.segment_audio(audio)

        # Extract features
        feature_segments = self.feature_extractor.extract_features(audio_segments)

        # Apply TPP to each feature segment
        tpp_vectors = []
        for features in feature_segments:
            tpp_vector = self.tpp.pool_features(features)
            tpp_vectors.append(tpp_vector.numpy())

        # Average the TPP vectors across segments
        if tpp_vectors:
            tpp_vector = np.mean(tpp_vectors, axis=0)
        else:
            tpp_vector = np.zeros(self.tpp_output_dim)

        return tpp_vector

    def build_vector_database(self, dataset: AudioDataset):
        """Build the vector database from training data."""
        vectors = []
        paths = []
        labels = []
        metadata = {'speaker_id': []}

        for idx in tqdm(range(len(dataset)), desc="Building vector database"):
            item = dataset[idx]
            audio_path = item['path']
            label = item['label']

            # Get speaker ID if available
            if 'metadata' in item and 'speaker_id' in item['metadata']:
                speaker_id = item['metadata']['speaker_id']
            else:
                speaker_id = "unknown"

            # Process audio
            tpp_vector = self.process_audio(audio_path)

            vectors.append(tpp_vector)
            paths.append(audio_path)
            labels.append(label)
            metadata['speaker_id'].append(speaker_id)

        # Convert lists to arrays
        vectors = np.vstack(vectors)

        # Add vectors to database
        self.vector_db.add_vectors(vectors, paths, labels, metadata)

        # Save the database
        self.vector_db.save()

    def retrieve_similar_vectors(self, query_vector: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve similar vectors from the database."""
        # Search for similar vectors
        distances, retrieved_labels = self.vector_db.search(query_vector)

        # Convert to PyTorch tensors
        distances_tensor = torch.tensor(distances, dtype=torch.float32, device=self.config.device)
        labels_tensor = torch.tensor(retrieved_labels, dtype=torch.float32, device=self.config.device)

        # Use distances to create embeddings tensor
        # Here we're using a simple approach - in a real system you might want to
        # retrieve the actual embeddings from a storage system
        similar_vectors = []
        for idx in range(len(distances)):
            # Get the index from the search results
            index = self.vector_db.index.reconstruct(idx)
            similar_vectors.append(index)

        similar_vectors = torch.tensor(np.array(similar_vectors), dtype=torch.float32, device=self.config.device)

        return similar_vectors, labels_tensor

    # Training and evaluation methods continue...
    def train(self, train_dataset: AudioDataset, val_dataset: Optional[AudioDataset] = None):
      """Train the detection model."""
      # Build vector database if not already built
      if self.vector_db.index is None:
          self.build_vector_database(train_dataset)

      # Set models to training mode
      self.projection_layer.train()
      self.detection_model.train()

      # Training loop
      best_val_loss = float('inf')
      patience_counter = 0

      for epoch in range(self.config.num_epochs):
          # Training phase
          self.projection_layer.train()
          self.detection_model.train()

          train_loss = 0.0
          train_correct = 0
          train_total = 0

          for idx in tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                item = train_dataset[idx]
                audio_path = item['path']
                label = item['label']

                # Process audio
                tpp_vector = self.process_audio(audio_path)

                # Retrieve similar vectors
                similar_vectors, _ = self.retrieve_similar_vectors(tpp_vector)

                # Create input tensor for projection layer
                input_embeddings = similar_vectors.unsqueeze(0)  # [1, top_k, embedding_dim]

                # Forward pass through projection layer
                projected = self.projection_layer(input_embeddings)

                # Forward pass through detection model
                output = self.detection_model(projected)

                # Convert label to tensor
                label_tensor = torch.tensor([label], dtype=torch.float32, device=self.config.device)

                # Calculate loss
                loss = self.criterion(output, label_tensor)

                # Backward pass and optimization
                self.projection_optimizer.zero_grad()
                self.detection_optimizer.zero_grad()
                loss.backward()
                self.projection_optimizer.step()
                self.detection_optimizer.step()

                # Update statistics
                train_loss += loss.item()
                train_correct += ((output > 0) == label_tensor).sum().item()
                train_total += 1

          train_loss /= train_total
          train_accuracy = train_correct / train_total

          # Validation phase
          if val_dataset:
              val_loss, val_accuracy = self.evaluate(val_dataset)
              print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                            f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
                            f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
              logging.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                            f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
                            f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

              # Early stopping check
              if val_loss < best_val_loss:
                  best_val_loss = val_loss
                  patience_counter = 0

                  # Save best model
                  self.save_models("best_model")
              else:
                  patience_counter += 1
                  if patience_counter >= self.config.early_stopping_patience:
                      logging.info(f"Early stopping triggered after {epoch+1} epochs")
                      break
          else:
              logging.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                            f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}")
              print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                            f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}")

      # Save final model
      self.save_models("final_model")

    def evaluate(self, val_dataset):
      """
      Evaluate the model on a validation dataset.

      Args:
          val_dataset: Validation dataset to evaluate on

      Returns:
          Tuple of (validation_loss, validation_accuracy)
      """
      # Set models to evaluation mode
      self.projection_layer.eval()
      self.detection_model.eval()

      val_loss = 0.0
      val_correct = 0
      val_total = 0

      # For advanced metrics calculation
      all_labels = []
      all_scores = []

      with torch.no_grad():
          for idx in tqdm(range(len(val_dataset)), desc="Evaluating"):
              item = val_dataset[idx]
              audio_path = item['path']
              label = item['label']

              try:
                  # Process audio
                  tpp_vector = self.process_audio(audio_path)

                  # Retrieve similar vectors
                  similar_vectors, _ = self.retrieve_similar_vectors(tpp_vector)

                  # Create input tensor for projection layer
                  input_embeddings = similar_vectors.unsqueeze(0)  # [1, top_k, embedding_dim]

                  # Forward pass through projection layer
                  projected = self.projection_layer(input_embeddings)

                  # Forward pass through detection model
                  output = self.detection_model(projected)

                  # Convert label to tensor
                  label_tensor = torch.tensor([label], dtype=torch.float32, device=self.config.device)

                  # Calculate loss
                  loss = self.criterion(output, label_tensor)

                  # Convert to probability with sigmoid
                  score = torch.sigmoid(output).item()

                  # Update statistics
                  val_loss += loss.item()
                  val_correct += ((output > 0) == label_tensor).sum().item()
                  val_total += 1

                  # Store for metrics calculation
                  all_labels.append(label)
                  all_scores.append(score)

              except Exception as e:
                  logging.error(f"Error evaluating sample {audio_path}: {str(e)}")
                  # Continue with next sample rather than crashing the evaluation
                  continue

      # Calculate average loss and accuracy
      if val_total > 0:
          val_loss /= val_total
          val_accuracy = val_correct / val_total
      else:
          logging.warning("No valid samples in validation set")
          val_loss = float('inf')
          val_accuracy = 0.0

      return val_loss, val_accuracy

    def evaluate_with_metrics(self, val_dataset):
      """
      Comprehensive evaluation with detailed metrics including EER.

      Args:
          val_dataset: Validation dataset to evaluate on

      Returns:
          Dictionary containing all evaluation metrics
      """
      # Get basic metrics first
      val_loss, val_accuracy = self.evaluate(val_dataset)

      # Rerun to collect scores for detailed metrics
      all_labels = []
      all_scores = []

      with torch.no_grad():
          for idx in tqdm(range(len(val_dataset)), desc="Computing metrics"):
              item = val_dataset[idx]
              audio_path = item['path']
              label = item['label']

              try:
                  # Process audio
                  tpp_vector = self.process_audio(audio_path)

                  # Retrieve similar vectors
                  similar_vectors, _ = self.retrieve_similar_vectors(tpp_vector)

                  # Forward passes
                  input_embeddings = similar_vectors.unsqueeze(0)
                  projected = self.projection_layer(input_embeddings)
                  output = self.detection_model(projected)

                  # Get probability score
                  score = torch.sigmoid(output).item()

                  # Store for metrics calculation
                  all_labels.append(label)
                  all_scores.append(score)

              except Exception as e:
                  logging.warning(f"Error processing {audio_path}: {str(e)}")
                  continue

      # Calculate detailed metrics
      metrics = {
          'loss': val_loss,
          'accuracy': val_accuracy,
      }

      # Compute EER and other metrics if we have enough samples
      if len(all_labels) > 1:
          try:
              from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

              # Convert to numpy arrays
              labels_np = np.array(all_labels)
              scores_np = np.array(all_scores)

              # Compute ROC curve and EER (Equal Error Rate)
              fpr, tpr, thresholds = roc_curve(labels_np, scores_np)
              fnr = 1 - tpr

              # Find the threshold where FPR = FNR (Equal Error Rate)
              eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
              eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

              # Calculate t-DCF (Tandem Detection Cost Function) if specified in config
              t_dcf = None
              if hasattr(self.config, 'p_target'):
                  # Using simplified t-DCF calculation
                  p_target = getattr(self.config, 'p_target', 0.5)
                  c_miss = getattr(self.config, 'c_miss', 1)
                  c_fa = getattr(self.config, 'c_fa', 1)

                  # Calculate normalized t-DCF
                  dcf = p_target * c_miss * fnr + (1 - p_target) * c_fa * fpr
                  min_dcf = np.min(dcf)
                  t_dcf = min_dcf / min(p_target * c_miss, (1 - p_target) * c_fa)

                  metrics['min_dcf'] = float(min_dcf)
                  metrics['t_dcf'] = float(t_dcf)

              # Area under ROC curve
              roc_auc = auc(fpr, tpr)

              # Store metrics
              metrics['eer'] = float(eer)
              metrics['eer_threshold'] = float(eer_threshold)
              metrics['roc_auc'] = float(roc_auc)

              # Compute confusion matrix at EER threshold
              y_pred = (scores_np >= eer_threshold).astype(int)
              cm = confusion_matrix(labels_np, y_pred)

              # Format confusion matrix for easier interpretation
              metrics['confusion_matrix'] = cm.tolist()

          except (ImportError, ValueError) as e:
              logging.warning(f"Could not compute detailed metrics: {str(e)}")

      return metrics

    def predict(self, audio_path: str) -> Dict:
        """Predict whether an audio file is bonafide or spoof."""
        # Set models to evaluation mode
        self.projection_layer.eval()
        self.detection_model.eval()

        with torch.no_grad():
            # Process audio
            tpp_vector = self.process_audio(audio_path)

            # Retrieve similar vectors
            similar_vectors, retrieved_labels = self.retrieve_similar_vectors(tpp_vector)

            # Create input tensor for projection layer
            input_embeddings = similar_vectors.unsqueeze(0)  # [1, top_k, embedding_dim]

            # Forward pass through projection layer
            projected = self.projection_layer(input_embeddings)

            # Forward pass through detection model
            output = self.detection_model(projected)

            # Convert to probability with sigmoid
            probability = torch.sigmoid(output).item()

            # Make prediction
            prediction = "bonafide" if probability >= 0.5 else "spoof"

        return {
            "prediction": prediction,
            "probability": probability,
            "retrieved_labels": retrieved_labels.cpu().numpy(),
        }

    def save_models(self, prefix: str = "model"):
        """Save the trained models."""
        models_dir = os.path.join(self.config.data_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Save projection layer
        torch.save(
            self.projection_layer.state_dict(),
            os.path.join(models_dir, f"{prefix}_projection.pt")
        )

        # Save detection model
        torch.save(
            self.detection_model.state_dict(),
            os.path.join(models_dir, f"{prefix}_detection.pt")
        )

        logging.info(f"Models saved with prefix '{prefix}'")

    def load_models(self, prefix: str = "best_model"):
        """Load the trained models."""
        models_dir = os.path.join(self.config.data_root, "models")

        # Load projection layer
        self.projection_layer.load_state_dict(
            torch.load(os.path.join(models_dir, f"{prefix}_projection.pt"))
        )

        # Load detection model
        self.detection_model.load_state_dict(
            torch.load(os.path.join(models_dir, f"{prefix}_detection.pt"))
        )

        logging.info(f"Models loaded with prefix '{prefix}'")
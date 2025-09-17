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

import os
import logging
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, List
import torch.nn as nn

class DeepfakeDetectionPipeline:
    """Main pipeline for audio deepfake detection with single-GPU optimizations."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # 1. Audio segmenter (CPU)
        self.audio_segmenter = AudioSegmenter(config)

        # 2. Feature extractor (kept on CPU internally, model lives on GPU)
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)

        # 3. Set feature_dim in config
        if not hasattr(config, "feature_dim"):
            config.feature_dim = self.feature_extractor.feature_dim
            print(f"Feature dimension set to: {config.feature_dim}")

        # 4. TPP and vector DB
        self.tpp = TemporalPyramidPooling(config)
        self.vector_db = VectorDatabase(config)

        # 5. Projection & detection (moved to GPU)
        self.projection_layer = ProjectionLayer(config, self.tpp.get_output_dim()).to(self.device)
        fuse_in_dim  = self.tpp.get_output_dim() + self.config.projection_output_dim
        fuse_out_dim = self.config.projection_output_dim
        self.fuse = nn.Linear(fuse_in_dim, fuse_out_dim).to(self.device)
        self.detection_model  = DetectionModel(config, fuse_out_dim).to(self.device)

        # 6. Optimizers & AMP scaler
        self.projection_optimizer = torch.optim.Adam(self.projection_layer.parameters(),
                                                     lr=config.learning_rate,
                                                     weight_decay=config.weight_decay)
        self.fuse_optimizer = torch.optim.Adam(self.fuse.parameters(),
                                       lr=config.learning_rate,
                                       weight_decay=config.weight_decay)
        self.detection_optimizer  = torch.optim.Adam(self.detection_model.parameters(),
                                                     lr=config.learning_rate,
                                                     weight_decay=config.weight_decay)

        self.scaler    = GradScaler("cuda")

        # Track training file IDs for leakage checks
        self.training_file_ids = set()


    def compute_pos_weight_from_dataset(self, dataset) -> float:
        from torch.utils.data import DataLoader
        import math, numpy as np, torch

        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
        pos = 0
        neg = 0
        for b in loader:
            y = b['label']
            if isinstance(y, torch.Tensor):
                y = y.float()
                pos += (y > 0.5).sum().item()
                neg += (y <= 0.5).sum().item()
            else:
                y = torch.tensor(y, dtype=torch.float32)
                pos += (y > 0.5).sum().item()
                neg += (y <= 0.5).sum().item()
        # Safe ratio
        pos_weight = (neg + 1.0) / (pos + 1.0)  # +1 smoothing to avoid div-by-zero
        if not np.isfinite(pos_weight):
            pos_weight = 1.0
        pos_weight = float(np.clip(pos_weight, 0.1, 10.0))  # clamp to reasonable range
        return pos_weight

    def process_audio_batch(self,
                            audio_paths: List[str],
                            audio_dataset: AudioDataset) -> torch.Tensor:
        """
        Batch-process audio files:
          - Load & segment on CPU
          - Extract features; move each tensor to GPU
          - Pool with TPP on GPU
        Returns a tensor of shape [batch_size, tpp_output_dim].
        """
        # 1. Load & segment
        segments_batch = []
        for path in audio_paths:
            wav = audio_dataset.load_audio(path)
            if wav is None:
                raise RuntimeError(f"Failed to load '{path}'")
            segments_batch.append(self.audio_segmenter.segment_audio(wav))

        # 2. Extract features and move to GPU
        #    extract_features returns List[Tensor(cpu)] per audio file
        feature_batches = []
        for segments in segments_batch:
            cpu_tensors = self.feature_extractor.extract_features(segments)
            # move each to GPU and keep batch dimension
            gpu_tensors = [t.to(self.device) for t in cpu_tensors]
            feature_batches.append(gpu_tensors)

        # 3. Pool with TPP (GPU); compute one vector per audio file
        pooled = []
        for gpu_list in feature_batches:
            # average TPP over all segments
            seg_pooled = [self.tpp.pool_features(ft) for ft in gpu_list]  # each ft on GPU
            mean_pooled = torch.mean(torch.stack(seg_pooled), dim=0)
            pooled.append(mean_pooled)

        return torch.stack(pooled)  # [batch, tpp_output_dim]


    def build_vector_database(self, train_dataset: AudioDataset):
        """Build the FAISS vector DB using batched GPU processing."""
        logging.info("Building vector database from TRAINING data...")
        loader = DataLoader(train_dataset,
                            batch_size=self.config.db_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=self.config.num_workers)

        vectors, paths, labels, metadata = [], [], [], {'speaker_id': []}
        self.training_file_ids.clear()
        audio_ds = AudioDataset(self.config, is_train=False)

        for batch in tqdm(loader, desc="Vector DB Build"):
            audio_paths = batch['path']
            batch_vecs = self.process_audio_batch(audio_paths, audio_ds).cpu().numpy()
            for vec, path, lbl, meta in zip(batch_vecs,
                                            audio_paths,
                                            batch['label'],
                                            batch['metadata']):
                file_id = os.path.basename(path)
                self.training_file_ids.add(file_id)
                vectors.append(vec)
                paths.append(path)
                labels.append(lbl)
                # metadata may be dict or string; handle both
                if isinstance(meta, dict):
                    speaker = meta.get('speaker_id', 'unknown')
                else:
                    speaker = meta
                metadata['speaker_id'].append(speaker)

        vectors = np.vstack(vectors)
        self.vector_db.add_vectors(vectors, paths, labels, metadata)
        self.vector_db.save()
        logging.info(f"Built vector DB ({len(vectors)} samples)")


    def retrieve_similar_vectors(self,query_vectors: torch.Tensor,
    query_paths: Optional[List[str]] = None,   # helps exclude only the query file(s)
    exclude_self: bool = True,
    return_info: bool = False,                 # when True, return neighbor paths
    return_distances: bool = False             # when True, return neighbor FAISS distances
    ):
      """
      Batched retrieval from FAISS.

      Returns:
          if return_info == False and return_distances == False:
              (vec_tensor [B,K,D], lbl_tensor [B,K])

          if return_info == True and return_distances == False:
              (vec_tensor [B,K,D], lbl_tensor [B,K], neighbor_paths: List[List[str]])

          if return_info == False and return_distances == True:
              (vec_tensor [B,K,D], lbl_tensor [B,K], dist_tensor [B,K])

          if both True:
              (vec_tensor [B,K,D], lbl_tensor [B,K], neighbor_paths, dist_tensor [B,K])
      """
      import numpy as np, os

      q_np = query_vectors.detach().cpu().numpy().astype(np.float32)  # [B, D]
      B = q_np.shape[0]
      K = int(self.config.top_k)
      D = self.tpp.get_output_dim()

      # Build exclusion set for exact files if provided
      exclude_ids = set()
      if exclude_self and query_paths is not None:
          exclude_ids = {os.path.basename(p) for p in query_paths}

      # Empty-index fallback
      if (self.vector_db.index is None) or (getattr(self.vector_db.index, "ntotal", 0) == 0):
          vec_tensor = torch.zeros(B, K, D, device=self.device)
          lbl_tensor = torch.zeros(B, K, device=self.device)
          empty_paths = [[""] * K for _ in range(B)]
          dist_tensor = torch.full((B, K), float("nan"), device=self.device)
          if return_info and return_distances:
              return vec_tensor, lbl_tensor, empty_paths, dist_tensor
          if return_info:
              return vec_tensor, lbl_tensor, empty_paths
          if return_distances:
              return vec_tensor, lbl_tensor, dist_tensor
          return vec_tensor, lbl_tensor

      # Over-retrieve to allow self-exclusion without falling below K
      k_search = K + (10 if exclude_self else 0)
      try:
          dists, idxs = self.vector_db.search_batch(q_np, k=k_search)  # dists/idxs: [B, k_search]
      except Exception:
          dists = np.zeros((B, 0), dtype=np.float32)
          idxs  = np.zeros((B, 0), dtype=np.int64)

      if idxs is None or idxs.ndim != 2 or idxs.shape[0] != B:
          idxs = np.zeros((B, 0), dtype=np.int64)
          dists = np.zeros((B, 0), dtype=np.float32)

      all_vecs, all_lbls, all_paths, all_dists = [], [], [], []

      for row_idx, (row_inds, row_dists) in enumerate(zip(idxs, dists)):
          chosen_vecs, chosen_lbls, chosen_paths, chosen_dists = [], [], [], []
          for ii, dd in zip(row_inds, row_dists):
              ii = int(ii)
              # Gather filename & apply self-exclusion
              fname = os.path.basename(self.vector_db.vector_paths[ii])
              if exclude_self:
                  # If we know the exact query file(s), exclude only those;
                  # otherwise fall back to training-file set as legacy behavior.
                  if query_paths is not None:
                      if fname in exclude_ids:
                          continue
                  else:
                      if fname in getattr(self, "training_file_ids", set()):
                          continue

              # Reconstruct vector + collect label/path/distance
              vec = self.vector_db.index.reconstruct(ii)
              chosen_vecs.append(vec)
              chosen_lbls.append(self.vector_db.vector_labels[ii])
              chosen_paths.append(self.vector_db.vector_paths[ii])
              chosen_dists.append(float(dd))

              if len(chosen_vecs) == K:
                  break

          # Pad if fewer than K
          while len(chosen_vecs) < K:
              chosen_vecs.append(np.zeros(D, dtype=np.float32))
              chosen_lbls.append(0.0)
              chosen_paths.append("")
              chosen_dists.append(float("nan"))

          all_vecs.append(chosen_vecs)
          all_lbls.append(chosen_lbls)
          all_paths.append(chosen_paths)
          all_dists.append(chosen_dists)

      vec_tensor = torch.as_tensor(np.stack(all_vecs, axis=0), device=self.device, dtype=torch.float32)
      lbl_tensor = torch.as_tensor(np.stack(all_lbls, axis=0), device=self.device, dtype=torch.float32)
      dist_tensor = torch.as_tensor(np.stack(all_dists, axis=0), device=self.device, dtype=torch.float32)

      if return_info and return_distances:
          return vec_tensor, lbl_tensor, all_paths, dist_tensor
      if return_info:
          return vec_tensor, lbl_tensor, all_paths
      if return_distances:
          return vec_tensor, lbl_tensor, dist_tensor
      return vec_tensor, lbl_tensor




    def train(self,
              train_dataset: AudioDataset,
              val_dataset: Optional[AudioDataset] = None):
        """Single-GPU, mixed-precision train loop."""
        if val_dataset:
            self.validate_no_leakage(val_dataset)
        if self.vector_db.index is None:
            self.build_vector_database(train_dataset)


        pos_weight = self.compute_pos_weight_from_dataset(train_dataset)
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device, dtype=torch.float32)
        )
        logging.info(f"Using pos_weight={pos_weight:.3f} for BCEWithLogitsLoss")
        loader = DataLoader(train_dataset,
                            batch_size=self.config.train_batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=self.config.num_workers)

        for epoch in range(self.config.num_epochs):
            self.projection_layer.train()
            self.detection_model.train()
            epoch_loss, correct, total = 0., 0, 0

            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                paths, lbls = batch['path'], batch['label']
                tpp = self.process_audio_batch(paths, train_dataset)  # [B, dim]
                vecs, _ = self.retrieve_similar_vectors(tpp, query_paths=paths, exclude_self=True)

                # ---- Sanity checks ----
                if torch.isnan(tpp).any():
                    raise RuntimeError("NaNs detected in TPP embeddings")
                if torch.isnan(vecs).any():
                    logging.warning("NaNs in retrieved neighbor vectors; replacing with zeros.")
                    vecs = torch.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)

                # Fraction of real (non-zero) neighbors to diagnose retrieval health
                with torch.no_grad():
                    nnz = (vecs.abs().sum(dim=-1) > 0).float().mean().item()
                    if nnz < 0.3:  # less than 30% of slots are real; training may stall
                        logging.warning(f"Low real-neighbor rate: {nnz:.2f}. Consider exclude_self=False temporarily.")

                with autocast("cuda"):
                    if vecs.ndim == 1:
                        # no retrieval neighbors? add a seq dim
                        vecs = vecs.unsqueeze(0).unsqueeze(1)
                    elif vecs.ndim == 2:
                        # no retrieval neighbors? add a seq dim
                        vecs = vecs.unsqueeze(1)
                    batch, seq_len, feat_dim = vecs.shape
                    assert seq_len > 0, f"Received seq_len=0, top_k={self.config.top_k}"
                    proj = self.projection_layer(vecs)
                    qvec = tpp                                      # [B, D_tpp]
                    fused = torch.cat([qvec, proj], dim=1)          # [B, D_tpp + D_proj]
                    fused = self.fuse(fused)                        # [B, D_proj]
                    logits = self.detection_model(fused)            # [B] or [B,1]
                    if logits.ndim == 1:
                      logits = logits.unsqueeze(-1)
                    labels = lbls.to(self.device).to(dtype=logits.dtype).view_as(logits)
                    loss = self.criterion(logits, labels)

                self.projection_optimizer.zero_grad()
                self.fuse_optimizer.zero_grad()
                self.detection_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                # Gradient clipping for stability
                self.scaler.unscale_(self.projection_optimizer)
                self.scaler.unscale_(self.fuse_optimizer)
                self.scaler.unscale_(self.detection_optimizer)
                torch.nn.utils.clip_grad_norm_(self.projection_layer.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.fuse.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.detection_model.parameters(), max_norm=1.0)
                self.scaler.step(self.projection_optimizer)
                self.scaler.step(self.fuse_optimizer)
                self.scaler.step(self.detection_optimizer)
                self.scaler.update()

                epoch_loss += loss.item() * lbls.size(0)
                preds = (logits > 0).to(labels.dtype)
                correct += (preds == labels).sum().item()
                total   += labels.numel()

            train_loss = epoch_loss / total
            train_acc  = correct / total

            if val_dataset:
                val_loss, val_acc = self.evaluate(val_dataset)
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc:{val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train {train_loss:.4f}/{train_acc:.4f}")

        self.save_models("final_model")


    def evaluate(self, val_dataset: AudioDataset) -> Tuple[float, float]:
        """Mixed-precision batched evaluation."""
        loader = DataLoader(val_dataset,
                            batch_size=self.config.eval_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=self.config.num_workers)
        self.projection_layer.eval()
        self.detection_model.eval()

        total_loss, correct, total = 0., 0, 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                paths, lbls = batch['path'], batch['label']
                tpp = self.process_audio_batch(paths, val_dataset)
                vecs, _ = self.retrieve_similar_vectors(tpp, query_paths=paths, exclude_self=True)
                with autocast("cuda"):
                    if vecs.ndim == 1:
                        # no retrieval neighbors? add a seq dim
                        vecs = vecs.unsqueeze(0).unsqueeze(1)
                    elif vecs.ndim == 2:
                        # no retrieval neighbors? add a seq dim
                        vecs = vecs.unsqueeze(1)
                    batch, seq_len, feat_dim = vecs.shape
                    assert seq_len > 0, f"Received seq_len=0, top_k={self.config.top_k}"
                    proj = self.projection_layer(vecs)
                    qvec = tpp
                    fused = torch.cat([qvec, proj], dim=1)
                    fused = self.fuse(fused)
                    logits = self.detection_model(fused)
                    if logits.ndim == 1:
                      logits = logits.unsqueeze(-1)
                    labels = lbls.to(self.device).to(dtype=logits.dtype).view_as(logits)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item() * lbls.size(0)
                preds = (logits > 0).to(labels.dtype)
                correct += (preds == labels).sum().item()
                total   += labels.numel()

        return total_loss / total, correct / total


    def _load_audio_for_inference(self, audio_path: str):
        import librosa
        sr = self.config.sample_rate
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        # pad to at least one segment
        min_len = int(self.config.segment_length * sr)
        if len(y) < min_len:
            import numpy as np
            y = np.pad(y, (0, min_len - len(y)))
        return y
    
    def predict(self, audio_path: str, threshold: float = 0.5):
        import torch, os, logging
        if (self.vector_db.index is None) or (getattr(self.vector_db.index, "ntotal", 0) == 0):
            logging.warning("Vector DB is empty; retrieval may be zero-padded.")

        self.projection_layer.eval()
        self.detection_model.eval()

        with torch.no_grad():
            # (A) LOAD RAW AUDIO, no AudioDataset → no meta.csv
            wav = self._load_audio_for_inference(audio_path)

            # (B) SEGMENT → FEATURES → TPP (same path as process_audio_batch)
            segments = self.audio_segmenter.segment_audio(wav)
            cpu_feats = self.feature_extractor.extract_features(segments)   # list of T(C,F) cpu tensors
            gpu_feats = [t.to(self.device) for t in cpu_feats]
            seg_vecs  = [self.tpp.pool_features(ft) for ft in gpu_feats]    # list of [D] gpu
            tpp_vec   = torch.mean(torch.stack(seg_vecs, dim=0), dim=0, keepdim=True)  # [1, D]

            # (C) RETRIEVE with paths + distances
            vecs, lbls, npaths, ndists = self.retrieve_similar_vectors(
                tpp_vec, query_paths=[audio_path], exclude_self=True,
                return_info=True, return_distances=True
            )
            if torch.count_nonzero(vecs) == 0:
                vecs, lbls, npaths, ndists = self.retrieve_similar_vectors(
                    tpp_vec, query_paths=[audio_path], exclude_self=False,
                    return_info=True, return_distances=True
                )

            # Ensure [1, K, D]
            if vecs.ndim == 2: vecs = vecs.unsqueeze(1)
            elif vecs.ndim == 1: vecs = vecs.unsqueeze(0).unsqueeze(1)

            # (D) FORWARD
            from torch.amp import autocast
            use_amp = getattr(self.config, "use_mixed_precision", False) and self.device.type == "cuda"
            ctx = autocast("cuda") if use_amp else torch.autocast("cpu", enabled=False)
            with ctx:
                proj   = self.projection_layer(vecs)     # [1, D_proj]
                fused  = torch.cat([tpp_vec, proj], dim=1)
                fused  = self.fuse(fused)
                logits = self.detection_model(fused)

            if logits.ndim == 1: logits = logits.unsqueeze(-1)
            prob  = torch.sigmoid(logits).detach().cpu().view(-1).mean().item()
            pred  = "spoof" if prob >= float(threshold) else "bona-fide"
            logit = logits.detach().cpu().view(-1).mean().item()

            # Build neighbor lists for the UI
            if isinstance(lbls, torch.Tensor):
                neigh_labels = [int(x) for x in lbls.squeeze(0).detach().cpu().tolist()]
            else:
                neigh_labels = list(lbls[0]) if isinstance(lbls, list) and len(lbls) else []
            neigh_paths = npaths[0] if isinstance(npaths, list) and len(npaths) else []
            if isinstance(ndists, torch.Tensor):
                neigh_dists = [float(x) for x in ndists.squeeze(0).detach().cpu().tolist()]
            else:
                neigh_dists = list(ndists[0]) if isinstance(ndists, list) and len(ndists) else []

            return {
                "prediction": pred,
                "probability": float(prob),
                "logit": float(logit),
                "retrieved_files": [os.path.basename(p) if p else "" for p in neigh_paths],
                "retrieved_labels": neigh_labels,
                "retrieved_distances": [
                    None if (isinstance(d, float) and d != d) else d for d in neigh_dists
                ],
                "retrieved": [
                    {"file": os.path.basename(p) if p else "", "path": p, "label": l, "distance": d}
                    for p, l, d in zip(neigh_paths, neigh_labels, neigh_dists)
                ],
            }


    def validate_no_leakage(self, val_dataset: AudioDataset):
        val_ids = {os.path.basename(item['path']) for item in val_dataset}
        overlap = self.training_file_ids & val_ids
        if overlap:
            raise ValueError(f"Data leakage! {len(overlap)} overlapping files.")
        logging.info("No data leakage detected.")


    def save_models(self, prefix: str = "model"):
        models_dir = os.path.join(self.config.data_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        torch.save(self.projection_layer.state_dict(),
                  os.path.join(models_dir, f"{prefix}_projection.pt"))
        torch.save(self.fuse.state_dict(),
                  os.path.join(models_dir, f"{prefix}_fuse.pt"))           # NEW
        torch.save(self.detection_model.state_dict(),
                  os.path.join(models_dir, f"{prefix}_detection.pt"))
        logging.info(f"Models saved under prefix '{prefix}'.")


    def load_models(self, prefix: str = "best_model"):
        import json, os, torch, logging
        models_dir = os.path.join(self.config.data_root, "models")

        # 1) Projection
        self.projection_layer.load_state_dict(
            torch.load(os.path.join(models_dir, f"{prefix}_projection.pt"),
                    map_location=self.device))

        # 2) Fuse (may not exist in old checkpoints)
        fuse_path = os.path.join(models_dir, f"{prefix}_fuse.pt")
        if os.path.exists(fuse_path):
            self.fuse.load_state_dict(torch.load(fuse_path, map_location=self.device))
        else:
            logging.warning("Fuse weights not found; using randomly initialized fuse layer.")

        # 3) Detection with BN/LN-robust load
        det_path = os.path.join(models_dir, f"{prefix}_detection.pt")
        sd = torch.load(det_path, map_location=self.device)

        def _try_load(strict: bool = True):
            try:
                compat = self.detection_model.load_state_dict(sd, strict=strict)
                # PyTorch returns IncompatibleKeys with missing/unexpected fields on strict=False
                if not strict:
                    logging.warning(f"DetectionModel load (strict=False) -> missing={compat.missing_keys}, "
                                    f"unexpected={compat.unexpected_keys}")
                return True
            except RuntimeError as e:
                logging.warning(f"DetectionModel load failed (strict={strict}): {e}")
                return False

        # try strict with current config
        if not _try_load(strict=True):
            msg = "missing running_mean"  # marker for BN buffer issues
            # If it smells like a BN mismatch, flip BN/LN and rebuild
            self.config.use_batch_norm = False
            self.config.use_layer_norm = True
            # Rebuild detection head with new flags
            fuse_out_dim = self.config.projection_output_dim
            self.detection_model = DetectionModel(self.config, fuse_out_dim).to(self.device)
            if not _try_load(strict=True):
                # final fallback – accept missing buffers/keys
                _try_load(strict=False)

        logging.info(f"Models loaded from prefix '{prefix}'.")

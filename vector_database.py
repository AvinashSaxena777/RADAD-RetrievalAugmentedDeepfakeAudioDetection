import logging
import os
from config import Config
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
class VectorDatabase:
    """GPU-optimized vector database for storing and retrieving feature vectors."""

    def __init__(self, config: Config):
        self.config = config
        self.index = None
        self.gpu_index = None
        self.vector_paths = []
        self.vector_labels = []
        self.vector_metadata = {}
        self.db_path = os.path.join(config.vector_db_path, "faiss_index.bin")
        self.metadata_path = os.path.join(config.vector_db_path, "metadata.pkl")

        # GPU resource management
        self.gpu_resources = None
        self.device_id = 0  # Single GPU setup

        # Create directory if it doesn't exist
        os.makedirs(config.vector_db_path, exist_ok=True)

        # Initialize GPU resources
        self._initialize_gpu_resources()

    def _initialize_gpu_resources(self):
      """Initialize GPU resources for FAISS operations (robust to FAISS build differences)."""
      try:
          import faiss, torch
          # Prefer torch to detect CUDA runtime; faiss.get_num_gpus() can still be 0 if faiss-gpu isn't installed
          if torch.cuda.is_available():
              self.device_id = torch.cuda.current_device()
              # IMPORTANT: assign to self.gpu_resources
              res = faiss.StandardGpuResources()
              # Optional scratch memory (safe across builds)
              if hasattr(res, "setTempMemory"):
                  # 512MB scratch is a good default for T4; avoid huge values on Colab
                  scratch = int(getattr(self.config, "faiss_temp_mem_bytes", 512 * 1024 * 1024))
                  res.setTempMemory(scratch)
              self.gpu_resources = res
              dev_name = torch.cuda.get_device_name(self.device_id)
              logging.info(f"FAISS GPU resources ready on device {self.device_id} ({dev_name})")
          else:
              self.gpu_resources = None
              logging.warning("CUDA not available; FAISS will use CPU.")
      except Exception as e:
          self.gpu_resources = None
          logging.warning(f"FAISS GPU init failed ({e}); falling back to CPU.")


    def create_index(self, dimension: int):
      import faiss
      index_type = self.config.vector_db_index_type.upper()

      # CPU baseline
      if index_type == "L2":
          cpu_index = faiss.IndexFlatL2(dimension)
      elif index_type == "IP":
          cpu_index = faiss.IndexFlatIP(dimension)
      elif index_type == "IVF":
          # IVF needs training
          nlist = int(getattr(self.config, "ivf_nlist", 4096))
          nlist = max(64, nlist)
          quantizer = faiss.IndexFlatL2(dimension)
          cpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
      else:
          raise ValueError(f"Unsupported index type: {index_type}")

      # Move to GPU if available
      if self.gpu_resources is not None:
          try:
              if index_type in ("L2", "IP"):
                  cfg = faiss.GpuIndexFlatConfig()
                  cfg.device = self.device_id
                  cfg.useFloat16 = bool(getattr(self.config, "use_float16", False))
                  if index_type == "L2":
                      self.index = faiss.GpuIndexFlatL2(self.gpu_resources, dimension, cfg)
                  else:
                      self.index = faiss.GpuIndexFlatIP(self.gpu_resources, dimension, cfg)
              else:
                  # IVF on GPU
                  self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.device_id, cpu_index)
              logging.info(f"Created FAISS index on GPU ({type(self.index).__name__}) dim={dimension}")
          except Exception as e:
              logging.warning(f"GPU index creation failed ({e}); using CPU index.")
              self.index = cpu_index
      else:
          self.index = cpu_index
          logging.info(f"Created FAISS CPU index ({type(self.index).__name__}) dim={dimension}")

      # Store a flag for cosine/IP usage
      self._cosine = (index_type == "IP") and bool(getattr(self.config, "normalize_for_ip", True))


    def _maybe_normalize(self, arr: np.ndarray) -> np.ndarray:
      if getattr(self, "_cosine", False):
          # L2-normalize rows
          norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
          arr = arr / norms
      return arr


    def add_vectors_batch(self, vectors: np.ndarray, paths: List[str], labels: List[int],
                      metadata: Dict, batch_size: int = 10000):
      if vectors.shape[0] == 0:
          logging.warning("No vectors to add to database")
          return

      if self.index is None:
          self.create_index(vectors.shape[1])

      # Normalize for cosine if using IP
      vectors = self._maybe_normalize(vectors.astype(np.float32, copy=False))
      vectors = np.ascontiguousarray(vectors)

      # If IVF, train once before adding
      try:
          import faiss
          if hasattr(self.index, "is_trained") and not self.index.is_trained:
              logging.info("Training IVF index...")
              # Use a representative subset for training to avoid long stalls
              train_n = min(50000, vectors.shape[0])
              self.index.train(vectors[:train_n])
      except Exception as e:
          logging.warning(f"Index training skipped/failed: {e}")

      total_vectors = vectors.shape[0]
      added = 0
      for start in range(0, total_vectors, batch_size):
          end = min(start + batch_size, total_vectors)
          batch = vectors[start:end]
          try:
              self.index.add(batch)  # GPU-accelerated if index is GPU
              added += batch.shape[0]
              # metadata bookkeeping
              self.vector_paths.extend(paths[start:end])
              self.vector_labels.extend(labels[start:end])
              for key, values in metadata.items():
                  self.vector_metadata.setdefault(key, [])
                  vals = values[start:end] if hasattr(values, '__getitem__') else [values] * len(batch)
                  self.vector_metadata[key].extend(vals)
          except Exception as e:
              logging.error(f"Error adding batch {start}-{end}: {e}")
              continue

      logging.info(f"Added {added}/{total_vectors} vectors. Index ntotal={self.index.ntotal}")


    def add_vectors(self, vectors: np.ndarray, paths: List[str], labels: List[int], metadata: Dict):
        """Add vectors to the database with automatic batching."""
        batch_size = getattr(self.config, 'vector_add_batch_size', 10000)
        self.add_vectors_batch(vectors, paths, labels, metadata, batch_size)

    def search_batch(self, query_vectors: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
      if self.index is None:
          raise ValueError("Vector database is empty. Build the database first.")

      k = int(k if k is not None else getattr(self.config, 'top_k', 5))
      if query_vectors.ndim == 1:
          query_vectors = query_vectors.reshape(1, -1)
      query_vectors = self._maybe_normalize(query_vectors.astype(np.float32, copy=False))
      query_vectors = np.ascontiguousarray(query_vectors)

      k = min(k, self.index.ntotal)
      if k <= 0:
          logging.warning("No vectors available for search")
          return np.zeros((len(query_vectors), 0), dtype=np.float32), np.zeros((len(query_vectors), 0), dtype=np.int64)

      # For IVF: tune nprobe
      try:
          if hasattr(self.index, 'nprobe') and hasattr(self.config, 'vector_db_nprobe'):
              self.index.nprobe = int(self.config.vector_db_nprobe)
      except Exception:
          pass

      distances, indices = self.index.search(query_vectors, k)
      return distances, indices


    def search(self, query_vector: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors (single query)."""
        distances, indices = self.search_batch(query_vector.reshape(1, -1), k)
        return distances[0] if len(distances) > 0 else np.array([]), indices[0] if len(indices) > 0 else np.array([])

    def save(self):
      try:
          import faiss, pickle
          if self.index is None:
              logging.warning("No index to save.")
              return
          # If GPU index, copy to CPU first
          to_save = self.index
          try:
              # Works if index is actually on GPU
              to_save = faiss.index_gpu_to_cpu(self.index)
          except Exception:
              pass
          faiss.write_index(to_save, self.db_path)

          meta = {
              'paths': self.vector_paths,
              'labels': self.vector_labels,
              'metadata': self.vector_metadata,
              'index_type': self.config.vector_db_index_type,
              'dimension': to_save.d if hasattr(to_save, 'd') else None
          }
          with open(self.metadata_path, 'wb') as f:
              pickle.dump(meta, f)
          logging.info(f"Saved FAISS index to {self.db_path} with {to_save.ntotal} vectors")
      except Exception as e:
          logging.error(f"Error saving vector database: {e}")

    def load(self):
      try:
          import faiss, pickle, os
          if not (os.path.exists(self.db_path) and os.path.exists(self.metadata_path)):
              logging.warning("No saved vector database found")
              return
          with open(self.metadata_path, 'rb') as f:
              meta = pickle.load(f)
          self.vector_paths   = meta['paths']
          self.vector_labels  = meta['labels']
          self.vector_metadata = meta['metadata']

          cpu_index = faiss.read_index(self.db_path)
          if self.gpu_resources is not None:
              try:
                  self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.device_id, cpu_index)
                  logging.info(f"Loaded FAISS index to GPU ({type(self.index).__name__}); ntotal={self.index.ntotal}")
              except Exception as e:
                  logging.warning(f"GPU load failed ({e}); using CPU index.")
                  self.index = cpu_index
          else:
              self.index = cpu_index
              logging.info(f"Loaded FAISS CPU index; ntotal={self.index.ntotal}")
      except Exception as e:
          logging.error(f"Error loading vector database: {e}")


    def get_gpu_memory_usage(self):
      """Return GPU memory usage via torch (reliable on Colab)."""
      try:
          import torch
          if torch.cuda.is_available():
              dev = self.device_id if hasattr(self, "device_id") else 0
              free, total = torch.cuda.mem_get_info(dev)
              used = total - free
              return {'used': int(used), 'total': int(total), 'utilization': float(used/total)}
      except Exception:
          pass
      return None


    def cleanup_gpu_resources(self):
        """Clean up GPU resources to prevent memory leaks."""
        if self.gpu_resources is not None:
            try:
                # Reset GPU resources
                del self.gpu_index
                self.gpu_index = None
                # Note: StandardGpuResources will be cleaned up by destructor
                logging.info("GPU resources cleaned up")
            except Exception as e:
                logging.warning(f"Error during GPU cleanup: {e}")

    def __del__(self):
        """Destructor to ensure GPU cleanup."""
        self.cleanup_gpu_resources()
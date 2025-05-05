import logging
import os
from config import Config
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
class VectorDatabase:
    """Vector database for storing and retrieving feature vectors."""
    
    def __init__(self, config: Config):
        self.config = config
        self.index = None
        self.vector_paths = []
        self.vector_labels = []
        self.vector_metadata = {}
        
        # Create directory for vector database if it doesn't exist
        os.makedirs(config.vector_db_path, exist_ok=True)
    
    def create_index(self, dimension: int):
        """Create a new FAISS index."""
        if self.config.vector_db_index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif self.config.vector_db_index_type == "IP":  # Inner Product
            self.index = faiss.IndexFlatIP(dimension)
        elif self.config.vector_db_index_type == "IVF":  # Inverted file index
            # Create a quantizer
            quantizer = faiss.IndexFlatL2(dimension)
            # Number of centroids (adjust based on dataset size)
            nlist = int(max(4, min(16, np.sqrt(len(self.vector_paths)))))
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            # Train with some vectors before use
        else:
            raise ValueError(f"Unsupported index type: {self.config.vector_db_index_type}")
    
    def add_vectors(self, vectors: np.ndarray, paths: List[str], labels: List[int], metadata: Dict = None):
        """Add vectors to the database."""
        if self.index is None:
            self.create_index(vectors.shape[1])
            
        # If index is IVF, train it first
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            self.index.train(vectors)
        
        # Add vectors to the index
        self.index.add(vectors)
        
        # Store paths and labels
        self.vector_paths.extend(paths)
        self.vector_labels.extend(labels)
        
        # Store metadata if provided
        if metadata:
            for key, values in metadata.items():
                if key not in self.vector_metadata:
                    self.vector_metadata[key] = []
                self.vector_metadata[key].extend(values)
    
    def search(self, query_vector: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors in the database."""
        if k is None:
            k = self.config.top_k
            
        if self.index is None:
            raise ValueError("Index is not initialized. Add vectors first.")
        
        # If index is IVF, set number of cells to visit
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = self.config.vector_db_nprobe
        
        # Reshape query vector if needed
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search for similar vectors
        distances, indices = self.index.search(query_vector, k)
        
        # Get labels of the retrieved vectors
        retrieved_labels = np.array([self.vector_labels[i] for i in indices[0]])
        
        return distances[0], retrieved_labels
    
    def save(self, filepath: str = None):
        """Save the vector database to disk."""
        if filepath is None:
            filepath = os.path.join(self.config.vector_db_path, "vector_db.pkl")
        
        # Save the index
        index_filepath = filepath.replace(".pkl", "_index.bin")
        faiss.write_index(self.index, index_filepath)
        
        # Save other data
        data = {
            'vector_paths': self.vector_paths,
            'vector_labels': self.vector_labels,
            'vector_metadata': self.vector_metadata,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logging.info(f"Vector database saved to {filepath}")
    
    def load(self, filepath: str = None):
        """Load the vector database from disk."""
        if filepath is None:
            filepath = os.path.join(self.config.vector_db_path, "vector_db.pkl")
        
        # Load the index
        index_filepath = filepath.replace(".pkl", "_index.bin")
        self.index = faiss.read_index(index_filepath)
        
        # Load other data
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vector_paths = data['vector_paths']
        self.vector_labels = data['vector_labels']
        self.vector_metadata = data['vector_metadata']
        
        logging.info(f"Vector database loaded from {filepath}")

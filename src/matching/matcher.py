"""Face matching with Faiss-based similarity search."""
import numpy as np
from typing import List, Tuple, Optional
import faiss
from pathlib import Path
import pickle


class FaceMatcher:
    """Face matcher using cosine similarity and Faiss index."""
    
    def __init__(self,
                 embedding_size: int = 512,
                 similarity_threshold: float = 0.6,
                 top_k: int = 5,
                 use_faiss: bool = True):
        """
        Initialize face matcher.
        
        Args:
            embedding_size: Size of embedding vectors
            similarity_threshold: Minimum similarity for positive match
            top_k: Number of top matches to return
            use_faiss: Use Faiss for fast similarity search
        """
        self.embedding_size = embedding_size
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.use_faiss = use_faiss
        
        # Initialize storage
        self.embeddings = []
        self.identities = []
        self.metadata = []
        
        # Initialize Faiss index
        if use_faiss:
            self._init_faiss_index()
    
    def _init_faiss_index(self):
        """Initialize Faiss index for similarity search."""
        # Use L2 distance (can convert to cosine similarity)
        # For cosine similarity: normalize embeddings and use L2
        self.index = faiss.IndexFlatL2(self.embedding_size)
        
        # Optional: use IVF index for larger databases
        # nlist = 100  # number of clusters
        # quantizer = faiss.IndexFlatL2(self.embedding_size)
        # self.index = faiss.IndexIVFFlat(quantizer, self.embedding_size, nlist)
        
        print(f"✓ Initialized Faiss index (dimension: {self.embedding_size})")
    
    def add_identity(self,
                    embedding: np.ndarray,
                    identity_id: str,
                    metadata: Optional[dict] = None):
        """
        Add identity to the gallery.
        
        Args:
            embedding: Face embedding vector
            identity_id: Unique identifier for the identity
            metadata: Additional metadata (name, image_path, etc.)
        """
        # Normalize embedding for cosine similarity
        embedding = self._normalize(embedding)
        
        # Add to storage
        self.embeddings.append(embedding)
        self.identities.append(identity_id)
        self.metadata.append(metadata or {})
        
        # Add to Faiss index
        if self.use_faiss:
            self.index.add(embedding.reshape(1, -1).astype('float32'))
        
        print(f"✓ Added identity: {identity_id}")
    
    def add_batch_identities(self,
                           embeddings: np.ndarray,
                           identity_ids: List[str],
                           metadata_list: Optional[List[dict]] = None):
        """Add multiple identities to gallery."""
        if metadata_list is None:
            metadata_list = [{}] * len(identity_ids)
        
        for embedding, identity_id, metadata in zip(embeddings, identity_ids, metadata_list):
            self.add_identity(embedding, identity_id, metadata)
    
    def match(self, query_embedding: np.ndarray) -> List[dict]:
        """
        Find matching identities for query embedding.
        
        Args:
            query_embedding: Query face embedding
            
        Returns:
            List of matches with identity, similarity, and metadata
        """
        if len(self.embeddings) == 0:
            return []
        
        # Normalize query embedding
        query_embedding = self._normalize(query_embedding)
        
        if self.use_faiss:
            return self._match_with_faiss(query_embedding)
        else:
            return self._match_with_numpy(query_embedding)
    
    def _match_with_faiss(self, query_embedding: np.ndarray) -> List[dict]:
        """Match using Faiss index."""
        # Search
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, min(self.top_k, len(self.embeddings)))
        
        # Convert L2 distances to cosine similarities
        # Since embeddings are normalized: cosine_sim = 1 - (L2_dist^2 / 2)
        similarities = 1 - (distances[0] ** 2 / 2)
        
        # Prepare results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity >= self.similarity_threshold:
                results.append({
                    'identity_id': self.identities[idx],
                    'similarity': float(similarity),
                    'distance': float(distances[0][list(indices[0]).index(idx)]),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def _match_with_numpy(self, query_embedding: np.ndarray) -> List[dict]:
        """Match using numpy cosine similarity."""
        embeddings_array = np.array(self.embeddings)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings_array, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= self.similarity_threshold:
                results.append({
                    'identity_id': self.identities[idx],
                    'similarity': float(similarity),
                    'distance': float(1 - similarity),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def remove_identity(self, identity_id: str) -> bool:
        """Remove identity from gallery."""
        try:
            idx = self.identities.index(identity_id)
            self.embeddings.pop(idx)
            self.identities.pop(idx)
            self.metadata.pop(idx)
            
            # Rebuild Faiss index
            if self.use_faiss:
                self._rebuild_faiss_index()
            
            print(f"✓ Removed identity: {identity_id}")
            return True
        except ValueError:
            print(f"⚠ Identity not found: {identity_id}")
            return False
    
    def _rebuild_faiss_index(self):
        """Rebuild Faiss index from current embeddings."""
        self._init_faiss_index()
        if len(self.embeddings) > 0:
            embeddings_array = np.array(self.embeddings).astype('float32')
            self.index.add(embeddings_array)
    
    def get_all_identities(self) -> List[dict]:
        """Get all identities in gallery."""
        return [
            {
                'identity_id': identity_id,
                'metadata': metadata
            }
            for identity_id, metadata in zip(self.identities, self.metadata)
        ]
    
    def save(self, filepath: str):
        """Save matcher state to file."""
        state = {
            'embeddings': self.embeddings,
            'identities': self.identities,
            'metadata': self.metadata,
            'embedding_size': self.embedding_size,
            'similarity_threshold': self.similarity_threshold,
            'top_k': self.top_k
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        # Save Faiss index separately
        if self.use_faiss:
            faiss_path = str(Path(filepath).with_suffix('.faiss'))
            faiss.write_index(self.index, faiss_path)
        
        print(f"✓ Saved matcher state to {filepath}")
    
    def load(self, filepath: str):
        """Load matcher state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.embeddings = state['embeddings']
        self.identities = state['identities']
        self.metadata = state['metadata']
        self.embedding_size = state['embedding_size']
        self.similarity_threshold = state['similarity_threshold']
        self.top_k = state['top_k']
        
        # Load Faiss index
        if self.use_faiss:
            faiss_path = str(Path(filepath).with_suffix('.faiss'))
            if Path(faiss_path).exists():
                self.index = faiss.read_index(faiss_path)
            else:
                self._rebuild_faiss_index()
        
        print(f"✓ Loaded matcher state from {filepath}")
    
    def benchmark_search(self, query_embedding: np.ndarray, num_runs: int = 1000) -> dict:
        """Benchmark search performance."""
        import time
        
        if len(self.embeddings) == 0:
            return {'error': 'No embeddings in gallery'}
        
        query_embedding = self._normalize(query_embedding)
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.match(query_embedding)
            times.append(time.time() - start)
        
        return {
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'searches_per_second': 1.0 / np.mean(times),
            'gallery_size': len(self.embeddings)
        }

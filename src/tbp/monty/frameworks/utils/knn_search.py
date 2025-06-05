# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import numpy as np
from scipy.spatial import KDTree

# Conditionally import FAISS if available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS is not available. GPU-based KNN search will not be available.")


class KNNSearchFactory:
    @staticmethod
    def create_index(backend='cpu', points=None, **kwargs):
        """Create a KNN index based on the specified backend.

        Args:
            backend: The backend to use for KNN search:
                - 'cpu': SciPy KDTree implementation (accurate, works on all platforms)
                - 'gpu': FAISS GPU implementation (faster for large datasets)
            points: Optional points to initialize the index with
            **kwargs: Additional arguments to pass to the index constructor
                For FAISS indices, you can specify:
                - nlist: Number of clusters for IVF indices (default: 1)
                - gpu_id: ID of GPU to use (default: 0)
                - batch_size: Batch size for large queries (None = auto)

        Returns:
            A KNNIndex instance
        """
        # Parse specific backend types for FAISS
        if backend.startswith('gpu_') and FAISS_AVAILABLE:
            index_type = backend[4:]  # Extract the index type after 'gpu_'
            return FAISSIndex(points=points, use_gpu=True, index_type=index_type, **kwargs)
        elif backend == 'gpu' and FAISS_AVAILABLE:
            return FAISSIndex(points=points, use_gpu=True, **kwargs)
        else:
            if backend != 'cpu' and backend != 'kdtree':
                logging.warning(f"Backend {backend} is not available or recognized. "
                              f"Falling back to CPU-based KDTree.")
            return KDTreeIndex(points=points, **kwargs)


class KNNIndex:
    """Base class for KNN search implementations."""
    def __init__(self, points=None, **kwargs):
        pass
    
    def build(self, points):
        """Build the index from points.
        
        Args:
            points: Points to build the index with (numpy array)
            
        Returns:
            self for method chaining
        """
        raise NotImplementedError
    
    def search(self, query_points, k, **kwargs):
        """Search for k nearest neighbors.
        
        Args:
            query_points: Points to search for (numpy array)
            k: Number of nearest neighbors to return
            **kwargs: Additional arguments for the search
            
        Returns:
            Tuple of (distances, indices) or just indices if return_distance=False
        """
        raise NotImplementedError


class KDTreeIndex(KNNIndex):
    """SciPy KDTree-based implementation."""
    def __init__(self, points=None, leafsize=40, **kwargs):
        """Initialize a KDTree-based KNN index.
        
        Args:
            points: Optional points to initialize the index with
            leafsize: Leaf size parameter for KDTree
            **kwargs: Additional arguments (ignored)
        """
        self.leafsize = leafsize
        self.index = None
        if points is not None:
            self.build(points)
    
    def build(self, points):
        """Build the KDTree index from points.
        
        Args:
            points: Points to build the index with (numpy array)
            
        Returns:
            self for method chaining
        """
        self.index = KDTree(points, leafsize=self.leafsize)
        return self
    
    def search(self, query_points, k, p=2, workers=1, return_distance=True):
        """Search for k nearest neighbors using KDTree.
        
        Args:
            query_points: Points to search for (numpy array)
            k: Number of nearest neighbors to return
            p: Distance metric (default: 2 for Euclidean)
            workers: Number of parallel jobs (default: 1)
            return_distance: Whether to return distances (default: True)
            
        Returns:
            Tuple of (distances, indices) or just indices if return_distance=False
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build() first.")
        
        distances, indices = self.index.query(
            query_points, 
            k=k, 
            p=p, 
            workers=workers
        )
        
        # Handle the case where k=1 by reshaping the output
        if k == 1:
            distances = np.expand_dims(distances, axis=1)
            indices = np.expand_dims(indices, axis=1)
        
        if return_distance:
            return distances, indices
        return indices


class FAISSIndex(KNNIndex):
    """FAISS-based implementation for GPU with enhanced performance options."""
    def __init__(self, points=None, use_gpu=True, index_type="flat", nlist=1, 
                gpu_id=0, batch_size=None, **kwargs):
        """Initialize a FAISS-based KNN index with configurable parameters.
        
        Args:
            points: Optional points to initialize the index with
            use_gpu: Whether to use GPU acceleration (default: True)
            nlist: Number of clusters for IVF indices (higher = more fine-grained)
            nprobe: Number of clusters to visit during search (higher = more accurate but slower)
            gpu_id: ID of GPU to use if multiple are available
            batch_size: Batch size for large queries (None = auto)
            **kwargs: Additional arguments (ignored)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Please install it first.")
        
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        # TODO investigate ANN with larger nlist
        self.nlist = nlist
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.index = None
        self.gpu_resources = None  # GPU resources
        self._index_size = 0  # Track index size for auto-tuning
        
        if points is not None:
            self.build(points)


    def __getstate__(self):
        """Custom serialization - handle GPU/CPU FAISS index conversion"""
        state = self.__dict__.copy()
        
        if self.index is not None:
            try:
                # Check if this is a GPU index and convert to CPU for serialization
                if self.use_gpu and hasattr(self.index, 'getDevice'):
                    # Convert GPU index to CPU for serialization
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    state['faiss_index_bytes'] = faiss.serialize_index(cpu_index)
                    state['was_gpu_index'] = True
                else:
                    # Already CPU index, serialize directly
                    state['faiss_index_bytes'] = faiss.serialize_index(self.index)
                    state['was_gpu_index'] = False
                    
            except RuntimeError as e:
                print(f"Failed to serialize FAISS index: {e}")
                state['faiss_index_bytes'] = None
                state['was_gpu_index'] = self.use_gpu
                
            # Remove unpicklable objects
            del state['index']
            if 'gpu_resources' in state:
                del state['gpu_resources']
        else:
            state['faiss_index_bytes'] = None
            state['was_gpu_index'] = False
            
        return state

    def __setstate__(self, state):
        """Custom deserialization - rebuild FAISS index and handle GPU conversion"""
        
        if state.get('faiss_index_bytes') is not None:
            # Deserialize the CPU index
            cpu_index = faiss.deserialize_index(state['faiss_index_bytes'])
            
            # If it was originally a GPU index and we want to use GPU, convert back
            if state.get('was_gpu_index', False) and self.use_gpu:
                try:
                    # Recreate GPU resources
                    self.gpu_resources = faiss.StandardGpuResources()
                    # Convert back to GPU
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, cpu_index)
                except Exception as e:
                    print(f"Failed to move index back to GPU: {e}. Using CPU instead.")
                    self.index = cpu_index
                    self.use_gpu = False
                    self.gpu_resources = None
            else:
                # Keep as CPU index
                self.index = cpu_index
                self.gpu_resources = None
        else:
            # No index to restore
            self.index = None
            self.gpu_resources = None
            
        # Clean up temporary state variables
        for key in ['faiss_index_bytes', 'was_gpu_index']:
            if key in state:
                del state[key]
                
        # Restore other attributes
        self.__dict__.update(state)
    
    def _create_cpu_index(self, d, n_points):
        """Create appropriate CPU index based on dimensionality and data size.
        
        Args:
            d: Dimensionality of the data
            n_points: Number of data points
            
        Returns:
            FAISS index
        """
            
        quantizer = faiss.IndexFlatL2(d)  # the other index
        cpu_index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
        return cpu_index
        
    
    def build(self, points):
        """Build the FAISS index from points with auto-tuning.
        
        Args:
            points: Points to build the index with (numpy array)
            
        Returns:
            self for method chaining
        """
        if points.size == 0:
            raise ValueError("Cannot build index with empty points array")
            
        # Ensure we're working with float32
        points = np.ascontiguousarray(points.astype('float32'))
        d = points.shape[1]  # dimensionality
        n_points = points.shape[0]  # number of points
        self._index_size = n_points
        
        # Create appropriate CPU index
        cpu_index = self._create_cpu_index(d, n_points)
        # if not cpu_index.is_trained:
        #     cpu_index.train(points)

        # try:
        #     # Check if GPU is available
        #     self.gpu_resources = faiss.StandardGpuResources()
        # except Exception as e:
        #     print(f"FAISS GPU not available: {e}")

        # self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, cpu_index)
        
        # Training is required for IVF indices
        # if isinstance(cpu_index, faiss.IndexIVF):
        if not cpu_index.is_trained:
            cpu_index.train(points)
            
        
        # Move to GPU if requested and available
        if self.use_gpu:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, cpu_index)
            except Exception as e:
                logging.warning(f"Error moving index to GPU: {e}. Falling back to CPU index.")
                self.index = cpu_index
                self.use_gpu = False
        else:
            self.index = cpu_index
        
        # Add points
        self.index.add(points)
        
        # # Set parameters for search
        # if hasattr(self.index, 'nprobe'):
        #     self.index.nprobe = min(self.nprobe, max(1, n_points // 10))
            
        return self
    
    def search(self, query_points, k, return_distance=True):
        """Search for k nearest neighbors using FAISS with batching for large queries.
        
        Args:
            query_points: Points to search for (numpy array)
            k: Number of nearest neighbors to return
            return_distance: Whether to return distances (default: True)
            
        Returns:
            Tuple of (distances, indices) or just indices if return_distance=False
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build() first.")
        
        # Ensure we're working with float32
        query_points = np.ascontiguousarray(query_points.astype('float32'))
        n_queries = query_points.shape[0]
        
        # For large query batches, split into smaller batches to avoid OOM errors
        # Default batch size is scaled with index size
        # if self.batch_size is None:
        #     # Heuristic: larger index = smaller batch size
        #     self.batch_size = max(1, min(1024, 10000 // max(1, self._index_size // 1000)))
        
        # # Process in batches if needed
        # if n_queries > self.batch_size:
        #     all_distances = []
        #     all_indices = []
            
        #     for i in range(0, n_queries, self.batch_size):
        #         batch = query_points[i:i+self.batch_size]
        #         batch_distances, batch_indices = self.index.search(batch, k)
                
        #         all_distances.append(batch_distances)
        #         all_indices.append(batch_indices)
                
        #     distances = np.vstack(all_distances)
        #     indices = np.vstack(all_indices)
        # else:
        #     # Search directly
        distances, indices = self.index.search(query_points, k)
        
        if return_distance:
            return distances, indices
        return indices
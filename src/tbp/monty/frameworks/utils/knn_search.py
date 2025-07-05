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
import threading
import time

import numpy as np
from scipy.spatial import KDTree

# Conditionally import FAISS if available
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS is not available.")

# Import profiler
from tbp.monty.frameworks.utils.knn_profiler import KNNProfiler

_SHARED_GPU_RESOURCES = None
_GPU_LOCK = threading.Lock()


def get_shared_gpu_resources():
    """Get the shared GPU resources, creating if accessed for the first time.

    Use this to prevent multiple instances of the GPU resource for multiple
    FAISS runtimes. All FAISS objects should use this shared resource to
    point to the same GPU.

    Returns:
        The shared FAISS GPU resource
    """
    global _SHARED_GPU_RESOURCES
    if _SHARED_GPU_RESOURCES is None:
        with _GPU_LOCK:
            if _SHARED_GPU_RESOURCES is None:
                try:
                    _SHARED_GPU_RESOURCES = faiss.StandardGpuResources()
                    print("Created shared GPU resources")
                except (RuntimeError, AttributeError) as e:
                    print(f"Failed to create GPU resources: {e}")
                    return None
    return _SHARED_GPU_RESOURCES


class KNNSearchFactory:
    @staticmethod
    def create_index(backend="cpu", points=None, **kwargs):
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
        if backend == "gpu" and FAISS_AVAILABLE:
            return FAISSIndex(points=points, use_gpu=True, **kwargs)
        else:
            if backend != "cpu" and backend != "kdtree":
                logging.warning(
                    f"Backend {backend} is not available or recognized. "
                    f"Falling back to CPU-based KDTree."
                )
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
        start_time = time.time()
        self.index = KDTree(points, leafsize=self.leafsize)
        elapsed_time = time.time() - start_time

        # Record profiling data
        profiler = KNNProfiler.get_instance()
        profiler.record_build("cpu", points.shape, elapsed_time)

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
        start_time = time.time()
        distances, indices = self.index.query(query_points, k=k, p=p, workers=workers)
        elapsed_time = time.time() - start_time

        # Record profiling data
        profiler = KNNProfiler.get_instance()
        profiler.record_search("cpu", len(query_points), k, elapsed_time)

        if return_distance:
            return distances, indices
        return indices


class FAISSIndex(KNNIndex):
    """FAISS-based implementation for GPU with enhanced performance options."""

    def __init__(
        self, points=None, use_gpu=True, nlist=1, gpu_id=0, batch_size=None, **kwargs
    ):
        """Initialize a FAISS-based KNN index with configurable parameters.

        Args:
            points: Optional points to initialize the index with
            use_gpu: Whether to use GPU acceleration (default: True)
            nlist: Number of clusters for IVF indices (higher = more fine-grained)
            gpu_id: ID of GPU to use if multiple are available
            batch_size: Batch size for large queries (None = auto)
            **kwargs: Additional arguments (ignored)

        Raises:
            ImportError: If FAISS is not available
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Please install it first.")

        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.nlist = nlist
        self.nprobe = nlist
        self.gpu_id = gpu_id
        max_gpu_batch_size = 65535
        self.batch_size = batch_size if batch_size else max_gpu_batch_size
        self.index = None
        self._index_size = 0  # Track index size for auto-tuning

        if points is not None:
            self.build(points)

    def __getstate__(self):
        """Custom serialization - handle GPU/CPU FAISS index conversion.

        Returns:
            Current serialized object state

        Raises:
            RuntimeError if serialization fails.
        """
        state = self.__dict__.copy()

        if self.index is not None:
            try:
                # Check if this is a GPU index and convert to CPU for serialization
                if self.use_gpu and hasattr(self.index, "getDevice"):
                    # Convert GPU index to CPU for serialization
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    state["faiss_index_bytes"] = faiss.serialize_index(cpu_index)
                    state["was_gpu_index"] = True
                else:
                    # Already CPU index, serialize directly
                    state["faiss_index_bytes"] = faiss.serialize_index(self.index)
                    state["was_gpu_index"] = False

            except RuntimeError as e:
                print(f"Failed to serialize FAISS index: {e}")
                state["faiss_index_bytes"] = None
                state["was_gpu_index"] = self.use_gpu

            # Remove unpicklable objects
            del state["index"]
            if "gpu_resources" in state:
                del state["gpu_resources"]
        else:
            state["faiss_index_bytes"] = None
            state["was_gpu_index"] = False

        return state

    def __setstate__(self, state):
        """Custom deserialization - rebuild FAISS index and handle GPU conversion.

        Args:
            state: Current object state to be deserialized
        """
        if state.get("faiss_index_bytes") is not None:
            # Deserialize the CPU index
            cpu_index = faiss.deserialize_index(state["faiss_index_bytes"])

            # If it was originally a GPU index and we want to use GPU, convert back
            if state.get("was_gpu_index", False) and self.use_gpu:
                try:
                    # Recreate GPU resources
                    shared_resources = get_shared_gpu_resources()
                    # Convert back to GPU
                    self.index = faiss.index_cpu_to_gpu(
                        shared_resources, self.gpu_id, cpu_index
                    )
                except (RuntimeError, AttributeError) as e:
                    print(f"Failed to move index back to GPU: {e}. Using CPU instead.")
                    self.index = cpu_index
                    self.use_gpu = False
            else:
                # Keep as CPU index
                self.index = cpu_index
        else:
            # No index to restore
            self.index = None

        # Clean up temporary state variables
        for key in ["faiss_index_bytes", "was_gpu_index"]:
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
        start_time = time.time()

        # Ensure we're working with float32
        points = np.ascontiguousarray(points.astype("float32"))
        d = points.shape[1]  # dimensionality
        n_points = points.shape[0]  # number of points
        self._index_size = n_points

        # Create appropriate CPU index
        cpu_index = self._create_cpu_index(d, n_points)

        # Training is required for IVF indices
        if not cpu_index.is_trained:
            cpu_index.train(points)

        # Move to GPU if requested and available
        if self.use_gpu:
            try:
                shared_resources = get_shared_gpu_resources()
                self.index = faiss.index_cpu_to_gpu(
                    shared_resources, self.gpu_id, cpu_index
                )
            except (RuntimeError, AttributeError) as e:
                logging.warning(f"Error moving index to GPU: {e}. Falling back to CPU.")
                self.index = cpu_index
                self.use_gpu = False
        else:
            self.index = cpu_index

        # Add points
        self.index.add(points)
        self.index.nprobe = self.nprobe

        elapsed_time = time.time() - start_time

        # Record profiling data
        profiler = KNNProfiler.get_instance()
        backend = "gpu" if self.use_gpu else "cpu"
        profiler.record_build(backend, points.shape, elapsed_time)

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
        # Ensure we're working with float32
        query_points = np.ascontiguousarray(query_points.astype("float32"))

        n_queries = query_points.shape[0]
        start_time = time.time()

        if n_queries > self.batch_size:
            all_distances = []
            all_indices = []

            for i in range(0, n_queries, self.batch_size):
                batch_end = min(i + self.batch_size, n_queries)
                batch = query_points[i:batch_end]

                batch_distances, batch_indices = self.index.search(batch, k)
                all_distances.append(batch_distances)
                all_indices.append(batch_indices)

            # Combine results
            distances = np.vstack(all_distances)
            indices = np.vstack(all_indices)
        else:
            # Small enough to search directly
            distances, indices = self.index.search(query_points, k)

        # Search directly
        elapsed_time = time.time() - start_time

        # Record profiling data
        profiler = KNNProfiler.get_instance()
        backend = "gpu" if self.use_gpu else "cpu"
        profiler.record_search(backend, query_points.shape, k, elapsed_time)

        if return_distance:
            return distances, indices
        return indices

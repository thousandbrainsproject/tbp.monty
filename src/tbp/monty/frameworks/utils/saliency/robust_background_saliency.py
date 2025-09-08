import math
import numpy as np
import networkx as nx
import scipy.spatial.distance
import skimage
import skimage.io
import skimage.color
from skimage.segmentation import slic
from skimage.util import img_as_float


class RobustBackgroundSaliency:
    """
    Robust Background Detection Saliency based on:
    "Saliency Optimization from Robust Background Detection" 
    by Wangjiang Zhu, Shuang Liang, Yichen Wei and Jian Sun, CVPR 2014
    
    This method uses superpixel segmentation and graph-based optimization
    to detect salient regions by identifying background regions first.
    """
    
    def __init__(self, n_segments=250, compactness=10, sigma=1, 
                 sigma_clr=10.0, sigma_bndcon=1.0, sigma_spa=0.25, mu=0.1):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.sigma_clr = sigma_clr
        self.sigma_bndcon = sigma_bndcon
        self.sigma_spa = sigma_spa
        self.mu = mu
    
    def _make_graph(self, grid):
        """Create graph from superpixel grid"""
        # get unique labels
        vertices = np.unique(grid)
     
        # map unique labels to [0,...,num_labels-1]
        reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
        grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
       
        # create edges
        down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
        right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
        all_edges = np.vstack([right, down])
        all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
        all_edges = np.sort(all_edges, axis=1)
        num_vertices = len(vertices)
        edge_hash = all_edges[:, 0] + num_vertices * all_edges[:, 1]
        # find unique connections
        edges = np.unique(edge_hash)
        # undo hashing
        edges = [[vertices[x % num_vertices],
                  vertices[x // num_vertices]] for x in edges] 
     
        return vertices, edges
    
    def _path_length(self, path, G):
        """Calculate path length in graph"""
        dist = 0.0
        for i in range(1, len(path)):
            dist += G[path[i - 1]][path[i]]['weight']
        return dist
    
    def _compute_saliency_cost(self, smoothness, w_bg, wCtr):
        """Solve the optimization problem for saliency"""
        n = len(w_bg)
        A = np.zeros((n, n))
        b = np.zeros(n)

        for x in range(n):
            A[x, x] = 2 * w_bg[x] + 2 * wCtr[x]
            b[x] = 2 * wCtr[x]
            for y in range(n):
                A[x, x] += 2 * smoothness[x, y]
                A[x, y] -= 2 * smoothness[x, y]
        
        x = np.linalg.solve(A, b)
        return x
    
    def compute_saliency(self, img):
        """
        Compute RBD saliency map for the given image.
        
        Args:
            img: Input image as numpy array (RGB format)
            
        Returns:
            sal: Saliency map normalized to [0, 255]
        """
        if len(img.shape) != 3:  # got a grayscale image
            img = skimage.color.gray2rgb(img)

        img_lab = img_as_float(skimage.color.rgb2lab(img))
        img_rgb = img_as_float(img)
        img_gray = img_as_float(skimage.color.rgb2gray(img))

        # Superpixel segmentation
        segments_slic = slic(img_rgb, n_segments=self.n_segments, 
                           compactness=self.compactness, sigma=self.sigma, 
                           enforce_connectivity=False)

        nrows, ncols = segments_slic.shape
        max_dist = math.sqrt(nrows * nrows + ncols * ncols)

        grid = segments_slic
        vertices, edges = self._make_graph(grid)

        gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]

        # Compute superpixel properties
        centers = {}
        colors = {}
        boundary = {}

        for v in vertices:
            centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
            colors[v] = np.mean(img_lab[grid == v], axis=0)

            x_pix = gridx[grid == v]
            y_pix = gridy[grid == v]

            if (np.any(x_pix == 0) or np.any(y_pix == 0) or 
                np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1)):
                boundary[v] = 1
            else:
                boundary[v] = 0

        # Build the graph
        G = nx.Graph()
        for edge in edges:
            pt1 = edge[0]
            pt2 = edge[1]
            color_distance = scipy.spatial.distance.euclidean(colors[pt1], colors[pt2])
            G.add_edge(pt1, pt2, weight=color_distance)

        # Add boundary connections
        for v1 in vertices:
            if boundary[v1] == 1:
                for v2 in vertices:
                    if boundary[v2] == 1:
                        color_distance = scipy.spatial.distance.euclidean(colors[v1], colors[v2])
                        G.add_edge(v1, v2, weight=color_distance)

        # Create vertex mapping for array indexing
        n_vertices = len(vertices)
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        
        # Compute distances and matrices
        geodesic = np.zeros((n_vertices, n_vertices), dtype=float)
        spatial = np.zeros((n_vertices, n_vertices), dtype=float)
        smoothness = np.zeros((n_vertices, n_vertices), dtype=float)
        adjacency = np.zeros((n_vertices, n_vertices), dtype=float)

        all_shortest_paths_color = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))

        for v1 in vertices:
            for v2 in vertices:
                i1, i2 = vertex_to_idx[v1], vertex_to_idx[v2]
                if v1 == v2:
                    geodesic[i1, i2] = 0
                    spatial[i1, i2] = 0
                    smoothness[i1, i2] = 0
                else:
                    geodesic[i1, i2] = self._path_length(all_shortest_paths_color[v1][v2], G)
                    spatial[i1, i2] = scipy.spatial.distance.euclidean(centers[v1], centers[v2]) / max_dist
                    smoothness[i1, i2] = (math.exp(-(geodesic[i1, i2] ** 2) / (2.0 * self.sigma_clr ** 2)) + 
                                        self.mu)

        # Set adjacency matrix
        for edge in edges:
            pt1 = edge[0]
            pt2 = edge[1]
            i1, i2 = vertex_to_idx[pt1], vertex_to_idx[pt2]
            adjacency[i1, i2] = 1
            adjacency[i2, i1] = 1

        for i1 in range(n_vertices):
            for i2 in range(n_vertices):
                smoothness[i1, i2] = adjacency[i1, i2] * smoothness[i1, i2]

        # Compute background and center weights
        area = {}
        len_bnd = {}
        bnd_con = {}
        w_bg = {}
        ctr = {}
        wCtr = {}

        for v1 in vertices:
            area[v1] = 0
            len_bnd[v1] = 0
            ctr[v1] = 0
            i1 = vertex_to_idx[v1]
            for v2 in vertices:
                i2 = vertex_to_idx[v2]
                d_app = geodesic[i1, i2]
                d_spa = spatial[i1, i2]
                w_spa = math.exp(-(d_spa ** 2) / (2.0 * self.sigma_spa ** 2))
                area_i = math.exp(-(geodesic[i1, i2] ** 2) / (2 * self.sigma_clr ** 2))
                area[v1] += area_i
                len_bnd[v1] += area_i * boundary[v2]
                ctr[v1] += d_app * w_spa
            bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])
            w_bg[v1] = 1.0 - math.exp(-(bnd_con[v1] ** 2) / (2 * self.sigma_bndcon ** 2))

        for v1 in vertices:
            wCtr[v1] = 0
            i1 = vertex_to_idx[v1]
            for v2 in vertices:
                i2 = vertex_to_idx[v2]
                d_app = geodesic[i1, i2]
                d_spa = spatial[i1, i2]
                w_spa = math.exp(-(d_spa ** 2) / (2.0 * self.sigma_spa ** 2))
                wCtr[v1] += d_app * w_spa * w_bg[v2]

        # Normalize wCtr values
        min_value = min(wCtr.values())
        max_value = max(wCtr.values())

        if max_value > min_value:
            for v in vertices:
                wCtr[v] = (wCtr[v] - min_value) / (max_value - min_value)
        else:
            for v in vertices:
                wCtr[v] = 0

        # Convert dictionaries to arrays for optimization
        w_bg_array = np.array([w_bg[v] for v in vertices])
        wCtr_array = np.array([wCtr[v] for v in vertices])
        
        # Solve optimization
        x = self._compute_saliency_cost(smoothness, w_bg_array, wCtr_array)

        # Create saliency map
        img_disp = img_gray.copy()
        for i, v in enumerate(vertices):
            img_disp[grid == v] = x[i]

        sal = img_disp.copy()
        sal_max = np.max(sal)
        sal_min = np.min(sal)
        
        if sal_max > sal_min:
            sal = 255 * ((sal - sal_min) / (sal_max - sal_min))
        else:
            sal = np.zeros_like(sal) * 255

        return sal.astype(np.uint8)
    
    def compute_saliency_from_path(self, img_path):
        """
        Compute saliency map from image file path.
        
        Args:
            img_path: Path to input image file
            
        Returns:
            sal: Saliency map normalized to [0, 255]
        """
        img = skimage.io.imread(img_path)
        return self.compute_saliency(img)
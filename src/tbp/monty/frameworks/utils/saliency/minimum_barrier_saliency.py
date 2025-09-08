import math
import numpy as np
import scipy.spatial.distance
import skimage
import skimage.io
import skimage.color
from skimage.util import img_as_float


class MinimumBarrierSaliency:
    """
    Minimum Barrier Distance Saliency based on:
    "Minimum Barrier Salient Object Detection at 80 FPS"
    
    This method computes saliency using minimum barrier distance and
    background detection based on border statistics.
    """
    
    def __init__(self, num_iters=3, method='b', border_ratio=0.1, alpha=50.0):
        self.num_iters = num_iters
        self.method = method  # 'b' for background method
        self.border_ratio = border_ratio
        self.alpha = alpha
    
    def _raster_scan(self, img, L, U, D):
        """Forward raster scan for MBD computation"""
        n_rows = len(img)
        n_cols = len(img[0])

        for x in range(1, n_rows - 1):
            for y in range(1, n_cols - 1):
                ix = img[x][y]
                d = D[x][y]

                u1 = U[x-1][y]
                l1 = L[x-1][y]

                u2 = U[x][y-1]
                l2 = L[x][y-1]

                b1 = max(u1, ix) - min(l1, ix)
                b2 = max(u2, ix) - min(l2, ix)

                if d <= b1 and d <= b2:
                    continue
                elif b1 < d and b1 <= b2:
                    D[x][y] = b1
                    U[x][y] = max(u1, ix)
                    L[x][y] = min(l1, ix)
                else:
                    D[x][y] = b2
                    U[x][y] = max(u2, ix)
                    L[x][y] = min(l2, ix)

        return True

    def _raster_scan_inv(self, img, L, U, D):
        """Backward raster scan for MBD computation"""
        n_rows = len(img)
        n_cols = len(img[0])

        for x in range(n_rows - 2, 0, -1):
            for y in range(n_cols - 2, 0, -1):
                ix = img[x][y]
                d = D[x][y]

                u1 = U[x+1][y]
                l1 = L[x+1][y]

                u2 = U[x][y+1]
                l2 = L[x][y+1]

                b1 = max(u1, ix) - min(l1, ix)
                b2 = max(u2, ix) - min(l2, ix)

                if d <= b1 and d <= b2:
                    continue
                elif b1 < d and b1 <= b2:
                    D[x][y] = b1
                    U[x][y] = max(u1, ix)
                    L[x][y] = min(l1, ix)
                else:
                    D[x][y] = b2
                    U[x][y] = max(u2, ix)
                    L[x][y] = min(l2, ix)

        return True

    def _mbd(self, img):
        """Compute Minimum Barrier Distance"""
        if len(img.shape) != 2:
            print('MBD requires 2D array')
            return None
        if img.shape[0] <= 3 or img.shape[1] <= 3:
            print('Image is too small for MBD')
            return None

        L = np.copy(img)
        U = np.copy(img)
        D = float('inf') * np.ones(img.shape)
        D[0, :] = 0
        D[-1, :] = 0
        D[:, 0] = 0
        D[:, -1] = 0

        # Convert to lists for faster iteration
        img_list = img.tolist()
        L_list = L.tolist()
        U_list = U.tolist()
        D_list = D.tolist()

        for x in range(self.num_iters):
            if x % 2 == 1:
                self._raster_scan(img_list, L_list, U_list, D_list)
            else:
                self._raster_scan_inv(img_list, L_list, U_list, D_list)

        return np.array(D_list)

    def _sigmoid_contrast(self, x):
        """Apply sigmoid function for contrast enhancement"""
        b = 10.0
        return 1.0 / (1.0 + math.exp(-b * (x - 0.5)))

    def compute_saliency(self, img):
        """
        Compute MBD saliency map for the given image.
        
        Args:
            img: Input image as numpy array (RGB format)
            
        Returns:
            sal: Saliency map normalized to [0, 255]
        """
        # Convert to grayscale for MBD computation
        img_mean = np.mean(img, axis=2)
        sal = self._mbd(img_mean)
        
        if sal is None:
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        if self.method == 'b':
            # Background detection method
            n_rows, n_cols, n_channels = img.shape
            img_size = math.sqrt(n_rows * n_cols)
            border_thickness = int(math.floor(self.border_ratio * img_size))

            img_lab = img_as_float(skimage.color.rgb2lab(img))
            
            # Extract border pixels
            px_left = img_lab[0:border_thickness, :, :]
            px_right = img_lab[n_rows - border_thickness:, :, :]
            px_top = img_lab[:, 0:border_thickness, :]
            px_bottom = img_lab[:, n_cols - border_thickness:, :]
            
            # Compute means
            px_mean_left = np.mean(px_left, axis=(0, 1))
            px_mean_right = np.mean(px_right, axis=(0, 1))
            px_mean_top = np.mean(px_top, axis=(0, 1))
            px_mean_bottom = np.mean(px_bottom, axis=(0, 1))

            # Reshape for covariance computation
            px_left = px_left.reshape((-1, 3))
            px_right = px_right.reshape((-1, 3))
            px_top = px_top.reshape((-1, 3))
            px_bottom = px_bottom.reshape((-1, 3))

            # Compute covariance matrices with regularization
            reg = 1e-6
            cov_left = np.linalg.inv(np.cov(px_left.T) + reg * np.eye(3))
            cov_right = np.linalg.inv(np.cov(px_right.T) + reg * np.eye(3))
            cov_top = np.linalg.inv(np.cov(px_top.T) + reg * np.eye(3))
            cov_bottom = np.linalg.inv(np.cov(px_bottom.T) + reg * np.eye(3))

            # Compute Mahalanobis distances
            img_lab_unrolled = img_lab.reshape(-1, 3)
            
            u_left = scipy.spatial.distance.cdist(img_lab_unrolled, px_mean_left.reshape(1, -1), 
                                                'mahalanobis', VI=cov_left)
            u_left = u_left.reshape((n_rows, n_cols))

            u_right = scipy.spatial.distance.cdist(img_lab_unrolled, px_mean_right.reshape(1, -1), 
                                                 'mahalanobis', VI=cov_right)
            u_right = u_right.reshape((n_rows, n_cols))

            u_top = scipy.spatial.distance.cdist(img_lab_unrolled, px_mean_top.reshape(1, -1), 
                                               'mahalanobis', VI=cov_top)
            u_top = u_top.reshape((n_rows, n_cols))

            u_bottom = scipy.spatial.distance.cdist(img_lab_unrolled, px_mean_bottom.reshape(1, -1), 
                                                   'mahalanobis', VI=cov_bottom)
            u_bottom = u_bottom.reshape((n_rows, n_cols))

            # Normalize distances
            u_left = u_left / np.max(u_left)
            u_right = u_right / np.max(u_right)
            u_top = u_top / np.max(u_top)
            u_bottom = u_bottom / np.max(u_bottom)

            # Combine background maps
            u_max = np.maximum(np.maximum(np.maximum(u_left, u_right), u_top), u_bottom)
            u_final = (u_left + u_right + u_top + u_bottom) - u_max

            # Combine with MBD result
            u_max_final = np.max(u_final)
            sal_max = np.max(sal)
            if sal_max > 0 and u_max_final > 0:
                sal = sal / sal_max + u_final / u_max_final

        # Apply center bias
        sal = sal / np.max(sal) if np.max(sal) > 0 else sal
        
        s = np.mean(sal)
        delta = self.alpha * math.sqrt(s)

        xv, yv = np.meshgrid(np.arange(sal.shape[1]), np.arange(sal.shape[0]))
        w, h = sal.shape
        w2 = w / 2.0
        h2 = h / 2.0

        C = 1 - np.sqrt(np.power(xv - h2, 2) + np.power(yv - w2, 2)) / math.sqrt(np.power(w2, 2) + np.power(h2, 2))
        sal = sal * C

        # Apply contrast enhancement
        sal = sal / np.max(sal) if np.max(sal) > 0 else sal
        
        # Vectorized sigmoid function
        sigmoid_v = np.vectorize(self._sigmoid_contrast)
        sal = sigmoid_v(sal)

        # Final normalization
        sal = sal * 255.0
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
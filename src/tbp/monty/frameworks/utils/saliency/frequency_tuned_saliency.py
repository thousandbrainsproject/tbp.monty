import numpy as np
import skimage.io
import skimage.color
from skimage import img_as_float
import scipy.signal


class FrequencyTunedSaliency:
    """
    Frequency-Tuned Saliency detection based on the method described in:
    "Frequency-tuned Salient Region Detection" by Achanta et al.
    
    This method computes saliency by comparing each pixel's color to the
    mean image color after applying Gaussian blur in LAB color space.
    """
    
    def __init__(self):
        # Gaussian kernel for blurring (5x1 separable filter)
        self.kernel_h = (1.0/16.0) * np.array([[1, 4, 6, 4, 1]])
        self.kernel_w = self.kernel_h.transpose()
    
    def compute_saliency(self, img):
        """
        Compute frequency-tuned saliency map for the given image.
        
        Args:
            img: Input image as numpy array (RGB format)
            
        Returns:
            sal: Saliency map normalized to [0, 255]
        """
        # Convert to float and LAB color space
        img_rgb = img_as_float(img)
        img_lab = skimage.color.rgb2lab(img_rgb)
        
        # Compute mean RGB values
        mean_val = np.mean(img_rgb, axis=(0, 1))
        
        # Apply Gaussian blur to each LAB channel separately
        blurred_l = scipy.signal.convolve2d(img_lab[:, :, 0], self.kernel_h, mode='same')
        blurred_a = scipy.signal.convolve2d(img_lab[:, :, 1], self.kernel_h, mode='same')
        blurred_b = scipy.signal.convolve2d(img_lab[:, :, 2], self.kernel_h, mode='same')
        
        blurred_l = scipy.signal.convolve2d(blurred_l, self.kernel_w, mode='same')
        blurred_a = scipy.signal.convolve2d(blurred_a, self.kernel_w, mode='same')
        blurred_b = scipy.signal.convolve2d(blurred_b, self.kernel_w, mode='same')
        
        # Stack the blurred LAB channels
        im_blurred = np.dstack([blurred_l, blurred_a, blurred_b])
        
        # Convert blurred LAB back to RGB for comparison with mean
        im_blurred_rgb = skimage.color.lab2rgb(im_blurred)
        
        # Compute saliency as Euclidean distance from mean color
        sal = np.linalg.norm(mean_val - im_blurred_rgb, axis=2)
        
        # Normalize to [0, 255]
        sal_max = np.max(sal)
        sal_min = np.min(sal)
        if sal_max > sal_min:
            sal = 255 * ((sal - sal_min) / (sal_max - sal_min))
        else:
            sal = np.zeros_like(sal)
            
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
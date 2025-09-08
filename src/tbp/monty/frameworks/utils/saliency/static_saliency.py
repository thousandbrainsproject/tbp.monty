"""
Saliency Method Comparison Tool

This module provides a comprehensive comparison of various saliency detection methods
including classical computer vision approaches and modern graph-based techniques.

Author: Refactored for better software design principles
"""

from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt

from frequency_tuned_saliency import FrequencyTunedSaliency
from robust_background_saliency import RobustBackgroundSaliency
from minimum_barrier_saliency import MinimumBarrierSaliency
from itti_koch_saliency import IttiKochSaliency


@dataclass
class SaliencyConfig:
    """Configuration settings for saliency comparison."""
    data_dir: Path
    input_pattern: str = "*.npy"
    output_suffix: str = "_saliency.png"
    figure_size: Tuple[int, int] = (15, 15)
    dpi: int = 150
    grid_shape: Tuple[int, int] = (3, 3)


class SaliencyMethod(ABC):
    """Abstract base class for all saliency methods."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of the saliency method."""
        pass
    
    @abstractmethod
    def compute_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Compute saliency map for the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Saliency map normalized to [0, 255] as uint8
        """
        pass


class OpenCVSaliencyMethod(SaliencyMethod):
    """Wrapper for OpenCV built-in saliency methods."""
    
    def __init__(self, method_name: str, opencv_creator_func):
        self._name = method_name
        self._creator_func = opencv_creator_func
        self._method = None
    
    @property
    def name(self) -> str:
        return self._name
    
    def compute_saliency(self, image: np.ndarray) -> np.ndarray:
        """Compute saliency using OpenCV method."""
        try:
            if self._method is None:
                self._method = self._creator_func()
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image
                
            success, saliency_map = self._method.computeSaliency(gray_image)
            
            if not success or saliency_map is None:
                logging.warning(f"{self.name} failed, returning zero map")
                return np.zeros_like(gray_image, dtype=np.uint8)
            
            return (saliency_map * 255).astype(np.uint8)
            
        except Exception as e:
            logging.error(f"Error in {self.name}: {e}")
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            return np.zeros_like(gray_image, dtype=np.uint8)


class CustomSaliencyMethod(SaliencyMethod):
    """Wrapper for custom saliency method implementations."""
    
    def __init__(self, name: str, saliency_instance, needs_bgr: bool = False):
        self._name = name
        self._instance = saliency_instance
        self._needs_bgr = needs_bgr
    
    @property
    def name(self) -> str:
        return self._name
    
    def compute_saliency(self, image: np.ndarray) -> np.ndarray:
        """Compute saliency using custom method."""
        try:
            # Convert color format if needed
            if self._needs_bgr and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            result = self._instance.compute_saliency(image)
            
            # Ensure output is uint8
            if result.dtype != np.uint8:
                result = result.astype(np.uint8)
                
            return result
            
        except Exception as e:
            logging.error(f"Error in {self.name}: {e}")
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            return np.zeros_like(gray_image, dtype=np.uint8)


class ImageLoader:
    """Handles loading and preprocessing of input images."""
    
    @staticmethod
    def load_numpy_image(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image from numpy file and prepare RGB and grayscale versions.
        
        Args:
            file_path: Path to the numpy image file
            
        Returns:
            Tuple of (rgb_image, grayscale_image)
        """
        try:
            img = np.load(file_path)
            logging.info(f"Loaded image: shape={img.shape}, dtype={img.dtype}, "
                        f"range=[{img.min():.3f}, {img.max():.3f}]")
            
            # Convert to RGB uint8
            img_rgb = (img[:, :, :3] * 255).astype(np.uint8)
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            return img_rgb, img_gray
            
        except Exception as e:
            logging.error(f"Failed to load image from {file_path}: {e}")
            raise


class SaliencyVisualizer:
    """Handles visualization of saliency comparison results."""
    
    def __init__(self, config: SaliencyConfig):
        self.config = config
    
    def create_comparison_plot(self, 
                             original_rgb: np.ndarray,
                             original_gray: np.ndarray,
                             saliency_results: Dict[str, np.ndarray],
                             save_path: Path) -> None:
        """
        Create and save a comparison plot of all saliency methods.
        
        Args:
            original_rgb: Original RGB image
            original_gray: Original grayscale image
            saliency_results: Dictionary mapping method names to saliency maps
            save_path: Path where to save the plot
        """
        try:
            rows, cols = self.config.grid_shape
            fig, axes = plt.subplots(rows, cols, figsize=self.config.figure_size)
            axes = axes.flatten()
            
            # First subplot: original RGB image
            axes[0].imshow(original_rgb)
            axes[0].set_title("Original RGB", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Second subplot: original grayscale image
            axes[1].imshow(original_gray, cmap='gray')
            axes[1].set_title("Original Grayscale", fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Remaining subplots: saliency maps
            for idx, (method_name, saliency_map) in enumerate(saliency_results.items(), 2):
                if idx < len(axes):
                    axes[idx].imshow(saliency_map, cmap='gray')
                    axes[idx].set_title(method_name, fontsize=12, fontweight='bold')
                    axes[idx].axis('off')
            
            # Hide any unused subplots
            for idx in range(len(saliency_results) + 2, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout(pad=2.0)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved comparison plot to {save_path}")
            
        except Exception as e:
            logging.error(f"Failed to create comparison plot: {e}")
            raise


class SaliencyComparator:
    """Main class that orchestrates the saliency comparison process."""
    
    def __init__(self, config: SaliencyConfig):
        self.config = config
        self.methods: List[SaliencyMethod] = []
        self.visualizer = SaliencyVisualizer(config)
        self._setup_logging()
        self._initialize_methods()
    
    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_methods(self) -> None:
        """Initialize all saliency detection methods."""
        try:
            # OpenCV methods
            self.methods.extend([
                OpenCVSaliencyMethod(
                    "Spectral Residual", 
                    cv2.saliency.StaticSaliencySpectralResidual_create
                ),
                OpenCVSaliencyMethod(
                    "Fine Grained", 
                    cv2.saliency.StaticSaliencyFineGrained_create
                ),
                OpenCVSaliencyMethod(
                    "ObjectnessBING", 
                    cv2.saliency.ObjectnessBING_create
                ),
            ])

            self.methods.extend([
                CustomSaliencyMethod("Frequency-Tuned", FrequencyTunedSaliency()),
                CustomSaliencyMethod("RBD Saliency", RobustBackgroundSaliency()),
                CustomSaliencyMethod("MBD Saliency", MinimumBarrierSaliency()),
                CustomSaliencyMethod("Itti-Koch", IttiKochSaliency(), needs_bgr=True),
            ])
            
            logging.info(f"Initialized {len(self.methods)} saliency methods")
            
        except Exception as e:
            logging.error(f"Failed to initialize methods: {e}")
            raise
    
    def run_comparison(self) -> None:
        """Execute the complete saliency comparison pipeline for all .npy files."""
        try:
            # Setup paths
            input_dir = (self.config.data_dir / "view_finder_images" / 
                        "view_finder_randrot" / "view_finder_rgbd" / "arrays")
            
            output_dir = (self.config.data_dir / "view_finder_images" / 
                         "view_finder_randrot" / "view_finder_rgbd" / "saliency")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all .npy files
            input_files = list(input_dir.glob(self.config.input_pattern))
            if not input_files:
                logging.warning(f"No files found matching pattern {self.config.input_pattern} in {input_dir}")
                return
            
            logging.info(f"Found {len(input_files)} files to process")
            
            # Process each file
            for input_path in sorted(input_files):
                try:
                    self._process_single_file(input_path, output_dir)
                except Exception as e:
                    logging.error(f"Failed to process {input_path.name}: {e}")
                    continue
            
            logging.info("Saliency comparison completed for all files")
            
        except Exception as e:
            logging.error(f"Comparison pipeline failed: {e}")
            raise
    
    def _process_single_file(self, input_path: Path, output_dir: Path) -> None:
        """Process a single .npy file."""
        # Create output filename
        output_filename = input_path.stem + self.config.output_suffix
        output_path = output_dir / output_filename
        
        # Load image
        logging.info(f"Processing {input_path.name}")
        img_rgb, img_gray = ImageLoader.load_numpy_image(input_path)
        
        # Compute saliency maps
        saliency_results = {}
        for method in self.methods:
            logging.info(f"  Computing saliency using {method.name}")
            saliency_map = method.compute_saliency(img_rgb)
            saliency_results[method.name] = saliency_map
        
        # Create visualization
        logging.info(f"  Creating visualization: {output_filename}")
        self.visualizer.create_comparison_plot(
            img_rgb, img_gray, saliency_results, output_path
        )
        
        logging.info(f"  Completed {input_path.name} -> {output_filename}")


def create_default_config() -> SaliencyConfig:
    """Create default configuration for saliency comparison."""
    data_dir = Path("~/tbp/results/dmc").expanduser()
    return SaliencyConfig(data_dir=data_dir)


def main() -> None:
    """Main entry point for the saliency comparison tool."""
    try:
        config = create_default_config()
        comparator = SaliencyComparator(config)
        comparator.run_comparison()
        
    except Exception as e:
        logging.error(f"Application failed: {e}")
        raise


if __name__ == "__main__":
    main()
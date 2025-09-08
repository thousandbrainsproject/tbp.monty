"""
Saliency detection utilities for Monty framework.

This module provides various saliency detection methods including:
- Frequency-Tuned Saliency (Achanta et al.)
- Robust Background Detection (Zhu et al.)  
- Minimum Barrier Distance (Zhang et al.)
- Itti-Koch Visual Attention Model
- OpenCV built-in methods (Spectral Residual, Fine Grained, ObjectnessBING)
"""

from .frequency_tuned_saliency import FrequencyTunedSaliency
from .robust_background_saliency import RobustBackgroundSaliency
from .minimum_barrier_saliency import MinimumBarrierSaliency
from .itti_koch_saliency import IttiKochSaliency

__all__ = [
    'FrequencyTunedSaliency',
    'RobustBackgroundSaliency', 
    'MinimumBarrierSaliency',
    'IttiKochSaliency'
]
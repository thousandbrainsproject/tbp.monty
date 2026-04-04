# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import numpy as np

__all__ = ["SDREncoder"]


class SDREncoder:
    """Convert real-valued CMP features to Sparse Distributed Representations.

    Uses random projection + top-k selection: multiply features by a fixed
    random matrix, take the k largest activations as active bits. This is
    fast, requires no learning, and preserves neighbourhood structure
    (similar inputs produce overlapping SDRs).

    Maintains separate projection matrices for morphological features (pose
    vectors: surface normal, curvature direction) and non-morphological
    features (HSV colour, curvature magnitudes). This separation is critical
    because morphology evidence can both add AND subtract from hypothesis
    evidence, while non-morphological feature evidence can only add.

    Attributes:
        sdr_dim: Total dimensionality of the output SDR.
        num_active: Number of active (1) bits in the output SDR.
        morph_dim: Number of bits allocated to morphological features.
        feat_dim: Number of bits allocated to non-morphological features.
    """

    # Morphological features: surface normal (3) + curvature direction (3) = 6
    MORPH_INPUT_DIM = 6
    # Default split: half the SDR for morphology, half for features
    DEFAULT_MORPH_FRACTION = 0.5

    def __init__(
        self,
        input_dim: int,
        sdr_dim: int = 2048,
        num_active: int = 41,
        morph_fraction: float = DEFAULT_MORPH_FRACTION,
        seed: int = 42,
    ):
        """Initialise the SDR encoder.

        Args:
            input_dim: Dimensionality of the non-morphological feature vector.
            sdr_dim: Total dimensionality of the output SDR.
            num_active: Total number of active bits in the output SDR.
            morph_fraction: Fraction of SDR bits allocated to morphological
                features. The remainder goes to non-morphological features.
            seed: Random seed for reproducible projections.
        """
        self.sdr_dim = sdr_dim
        self.num_active = num_active

        # Split SDR dimensions and active bits between morph and non-morph
        self.morph_dim = int(sdr_dim * morph_fraction)
        self.feat_dim = sdr_dim - self.morph_dim
        self.morph_active = int(num_active * morph_fraction)
        self.feat_active = num_active - self.morph_active

        rng = np.random.default_rng(seed)

        # Separate projection matrices for morph and non-morph features
        self._morph_projection = rng.standard_normal(
            (self.morph_dim, self.MORPH_INPUT_DIM)
        )
        self._feat_projection = rng.standard_normal(
            (self.feat_dim, input_dim)
        )

    def encode(
        self,
        morphological_features: np.ndarray | None,
        non_morphological_features: np.ndarray | None,
    ) -> np.ndarray:
        """Encode both morphological and non-morphological features into one SDR.

        The output SDR is the concatenation of separately encoded morph and
        non-morph sub-SDRs. This preserves the ability to compute evidence
        from each independently.

        Args:
            morphological_features: Flattened pose vector features (surface
                normal + curvature direction, shape (6,)). Can be None if
                pose information is unavailable.
            non_morphological_features: Non-pose features (HSV, curvature
                magnitudes, etc.). Can be None if features are unavailable.

        Returns:
            Binary SDR of shape (sdr_dim,) with exactly num_active bits set.
        """
        morph_sdr = self.encode_morphological(morphological_features)
        feat_sdr = self.encode_non_morphological(non_morphological_features)
        return np.concatenate([morph_sdr, feat_sdr])

    def encode_morphological(
        self,
        features: np.ndarray | None,
    ) -> np.ndarray:
        """Encode morphological features (pose vectors) into an SDR.

        Args:
            features: Flattened pose vectors of shape (6,): surface normal
                (3) + principal curvature direction (3). None produces
                an all-zero SDR.

        Returns:
            Binary SDR of shape (morph_dim,) with morph_active bits set
            (or all zeros if features is None).
        """
        sdr = np.zeros(self.morph_dim, dtype=np.float64)
        if features is None or self.morph_active == 0:
            return sdr
        projected = self._morph_projection @ features
        top_k = np.argpartition(projected, -self.morph_active)[-self.morph_active:]
        sdr[top_k] = 1.0
        return sdr

    def encode_non_morphological(
        self,
        features: np.ndarray | None,
    ) -> np.ndarray:
        """Encode non-morphological features into an SDR.

        Args:
            features: Feature vector (HSV, curvatures, etc.). None produces
                an all-zero SDR.

        Returns:
            Binary SDR of shape (feat_dim,) with feat_active bits set
            (or all zeros if features is None).
        """
        sdr = np.zeros(self.feat_dim, dtype=np.float64)
        if features is None or self.feat_active == 0:
            return sdr
        projected = self._feat_projection @ features
        top_k = np.argpartition(projected, -self.feat_active)[-self.feat_active:]
        sdr[top_k] = 1.0
        return sdr

    @staticmethod
    def sdr_overlap(a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalised overlap between two SDRs.

        Returns the fraction of active bits in a that are also active in b,
        normalised by the number of active bits in a. Returns 0 if a has
        no active bits.

        Args:
            a: First SDR (binary vector).
            b: Second SDR (binary vector).

        Returns:
            Overlap score in [0, 1].
        """
        a_active = np.sum(a > 0.5)
        if a_active == 0:
            return 0.0
        return float(np.sum((a > 0.5) & (b > 0.5)) / a_active)

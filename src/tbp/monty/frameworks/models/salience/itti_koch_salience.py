# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import cv2
import numpy as np

from .strategies import SalienceStrategy


class IttiKochSalienceStrategy(SalienceStrategy):
    """Implementation of the Itti-Koch visual attention model.

    Based on "A model of saliency-based visual attention for rapid scene analysis"
    by L. Itti, C. Koch, and E. Niebur, IEEE TPAMI 1998.

    This model computes saliency using intensity, color, and orientation conspicuity maps
    through center-surround operations on multi-scale image pyramids.
    """

    def __init__(
        self,
        width=None,
        height=None,
        weight_intensity=0.33,
        weight_color=0.33,
        weight_orientation=0.33,
        default_step_local=16,
        pyramid_levels=9,
    ):
        self.width = width
        self.height = height
        self.weight_intensity = weight_intensity
        self.weight_color = weight_color
        self.weight_orientation = weight_orientation
        self.default_step_local = default_step_local
        self.pyramid_levels = pyramid_levels
        self.prev_frame = None
        self.SM = None

        # Validate pyramid levels for center-surround operations
        if pyramid_levels < 7:
            raise ValueError(
                "pyramid_levels must be at least 7 for center-surround operations (need levels 2-6)"
            )

        # Gabor kernels for orientation detection
        self.GaborKernel0 = np.array(
            [
                [
                    1.85212e-06,
                    1.28181e-05,
                    -0.000350433,
                    -0.000136537,
                    0.002010422,
                    -0.000136537,
                    -0.000350433,
                    1.28181e-05,
                    1.85212e-06,
                ],
                [
                    2.80209e-05,
                    0.000193926,
                    -0.005301717,
                    -0.002065674,
                    0.030415784,
                    -0.002065674,
                    -0.005301717,
                    0.000193926,
                    2.80209e-05,
                ],
                [
                    0.000195076,
                    0.001350077,
                    -0.036909595,
                    -0.014380852,
                    0.211749204,
                    -0.014380852,
                    -0.036909595,
                    0.001350077,
                    0.000195076,
                ],
                [
                    0.000624940,
                    0.004325061,
                    -0.118242318,
                    -0.046070008,
                    0.678352526,
                    -0.046070008,
                    -0.118242318,
                    0.004325061,
                    0.000624940,
                ],
                [
                    0.000921261,
                    0.006375831,
                    -0.174308068,
                    -0.067914552,
                    1.000000000,
                    -0.067914552,
                    -0.174308068,
                    0.006375831,
                    0.000921261,
                ],
                [
                    0.000624940,
                    0.004325061,
                    -0.118242318,
                    -0.046070008,
                    0.678352526,
                    -0.046070008,
                    -0.118242318,
                    0.004325061,
                    0.000624940,
                ],
                [
                    0.000195076,
                    0.001350077,
                    -0.036909595,
                    -0.014380852,
                    0.211749204,
                    -0.014380852,
                    -0.036909595,
                    0.001350077,
                    0.000195076,
                ],
                [
                    2.80209e-05,
                    0.000193926,
                    -0.005301717,
                    -0.002065674,
                    0.030415784,
                    -0.002065674,
                    -0.005301717,
                    0.000193926,
                    2.80209e-05,
                ],
                [
                    1.85212e-06,
                    1.28181e-05,
                    -0.000350433,
                    -0.000136537,
                    0.002010422,
                    -0.000136537,
                    -0.000350433,
                    1.28181e-05,
                    1.85212e-06,
                ],
            ],
            dtype=np.float32,
        )

        self.GaborKernel45 = np.array(
            [
                [
                    4.04180e-06,
                    2.25320e-05,
                    -0.000279806,
                    -0.001028923,
                    3.79931e-05,
                    0.000744712,
                    0.000132863,
                    -9.04408e-06,
                    -1.01551e-06,
                ],
                [
                    2.25320e-05,
                    0.000925120,
                    0.002373205,
                    -0.013561362,
                    -0.022947700,
                    0.000389916,
                    0.003516954,
                    0.000288732,
                    -9.04408e-06,
                ],
                [
                    -0.000279806,
                    0.002373205,
                    0.044837725,
                    0.052928748,
                    -0.139178011,
                    -0.108372072,
                    0.000847346,
                    0.003516954,
                    0.000132863,
                ],
                [
                    -0.001028923,
                    -0.013561362,
                    0.052928748,
                    0.460162150,
                    0.249959607,
                    -0.302454279,
                    -0.108372072,
                    0.000389916,
                    0.000744712,
                ],
                [
                    3.79931e-05,
                    -0.022947700,
                    -0.139178011,
                    0.249959607,
                    1.000000000,
                    0.249959607,
                    -0.139178011,
                    -0.022947700,
                    3.79931e-05,
                ],
                [
                    0.000744712,
                    0.003899160,
                    -0.108372072,
                    -0.302454279,
                    0.249959607,
                    0.460162150,
                    0.052928748,
                    -0.013561362,
                    -0.001028923,
                ],
                [
                    0.000132863,
                    0.003516954,
                    0.000847346,
                    -0.108372072,
                    -0.139178011,
                    0.052928748,
                    0.044837725,
                    0.002373205,
                    -0.000279806,
                ],
                [
                    -9.04408e-06,
                    0.000288732,
                    0.003516954,
                    0.000389916,
                    -0.022947700,
                    -0.013561362,
                    0.002373205,
                    0.000925120,
                    2.25320e-05,
                ],
                [
                    -1.01551e-06,
                    -9.04408e-06,
                    0.000132863,
                    0.000744712,
                    3.79931e-05,
                    -0.001028923,
                    -0.000279806,
                    2.25320e-05,
                    4.04180e-06,
                ],
            ],
            dtype=np.float32,
        )

        self.GaborKernel90 = np.array(
            [
                [
                    1.85212e-06,
                    2.80209e-05,
                    0.000195076,
                    0.000624940,
                    0.000921261,
                    0.000624940,
                    0.000195076,
                    2.80209e-05,
                    1.85212e-06,
                ],
                [
                    1.28181e-05,
                    0.000193926,
                    0.001350077,
                    0.004325061,
                    0.006375831,
                    0.004325061,
                    0.001350077,
                    0.000193926,
                    1.28181e-05,
                ],
                [
                    -0.000350433,
                    -0.005301717,
                    -0.036909595,
                    -0.118242318,
                    -0.174308068,
                    -0.118242318,
                    -0.036909595,
                    -0.005301717,
                    -0.000350433,
                ],
                [
                    -0.000136537,
                    -0.002065674,
                    -0.014380852,
                    -0.046070008,
                    -0.067914552,
                    -0.046070008,
                    -0.014380852,
                    -0.002065674,
                    -0.000136537,
                ],
                [
                    0.002010422,
                    0.030415784,
                    0.211749204,
                    0.678352526,
                    1.000000000,
                    0.678352526,
                    0.211749204,
                    0.030415784,
                    0.002010422,
                ],
                [
                    -0.000136537,
                    -0.002065674,
                    -0.014380852,
                    -0.046070008,
                    -0.067914552,
                    -0.046070008,
                    -0.014380852,
                    -0.002065674,
                    -0.000136537,
                ],
                [
                    -0.000350433,
                    -0.005301717,
                    -0.036909595,
                    -0.118242318,
                    -0.174308068,
                    -0.118242318,
                    -0.036909595,
                    -0.005301717,
                    -0.000350433,
                ],
                [
                    1.28181e-05,
                    0.000193926,
                    0.001350077,
                    0.004325061,
                    0.006375831,
                    0.004325061,
                    0.001350077,
                    0.000193926,
                    1.28181e-05,
                ],
                [
                    1.85212e-06,
                    2.80209e-05,
                    0.000195076,
                    0.000624940,
                    0.000921261,
                    0.000624940,
                    0.000195076,
                    2.80209e-05,
                    1.85212e-06,
                ],
            ],
            dtype=np.float32,
        )

        self.GaborKernel135 = np.array(
            [
                [
                    -1.01551e-06,
                    -9.04408e-06,
                    0.000132863,
                    0.000744712,
                    3.79931e-05,
                    -0.001028923,
                    -0.000279806,
                    2.2532e-05,
                    4.0418e-06,
                ],
                [
                    -9.04408e-06,
                    0.000288732,
                    0.003516954,
                    0.000389916,
                    -0.022947700,
                    -0.013561362,
                    0.002373205,
                    0.00092512,
                    2.2532e-05,
                ],
                [
                    0.000132863,
                    0.003516954,
                    0.000847346,
                    -0.108372072,
                    -0.139178011,
                    0.052928748,
                    0.044837725,
                    0.002373205,
                    -0.000279806,
                ],
                [
                    0.000744712,
                    0.000389916,
                    -0.108372072,
                    -0.302454279,
                    0.249959607,
                    0.46016215,
                    0.052928748,
                    -0.013561362,
                    -0.001028923,
                ],
                [
                    3.79931e-05,
                    -0.022947700,
                    -0.139178011,
                    0.249959607,
                    1.000000000,
                    0.249959607,
                    -0.139178011,
                    -0.0229477,
                    3.79931e-05,
                ],
                [
                    -0.001028923,
                    -0.013561362,
                    0.052928748,
                    0.460162150,
                    0.249959607,
                    -0.302454279,
                    -0.108372072,
                    0.000389916,
                    0.000744712,
                ],
                [
                    -0.000279806,
                    0.002373205,
                    0.044837725,
                    0.052928748,
                    -0.139178011,
                    -0.108372072,
                    0.000847346,
                    0.003516954,
                    0.000132863,
                ],
                [
                    2.25320e-05,
                    0.000925120,
                    0.002373205,
                    -0.013561362,
                    -0.022947700,
                    0.000389916,
                    0.003516954,
                    0.000288732,
                    -9.04408e-06,
                ],
                [
                    4.04180e-06,
                    2.25320e-05,
                    -0.000279806,
                    -0.001028923,
                    3.79931e-05,
                    0.000744712,
                    0.000132863,
                    -9.04408e-06,
                    -1.01551e-06,
                ],
            ],
            dtype=np.float32,
        )

    def _extract_RGBI(self, input_image):
        """Extract R, G, B, and intensity channels"""
        src = np.float32(input_image) * 1.0 / 255
        (B, G, R) = cv2.split(src)
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return R, G, B, I

    def _create_gaussian_pyramid(self, src):
        """Create Gaussian pyramid with configurable levels"""
        dst = [src]
        for i in range(1, self.pyramid_levels):
            now_dst = cv2.pyrDown(dst[i - 1])
            dst.append(now_dst)
        return dst

    def _center_surround_diff(self, gaussian_maps):
        """Compute center-surround differences with configurable pyramid levels"""
        dst = []
        # Center levels: 2, 3, 4 (original implementation)
        # Surround levels: center + 3, center + 4
        max_center_level = min(
            4, self.pyramid_levels - 5
        )  # Ensure we have enough levels for surround

        for s in range(2, max_center_level + 1):
            if s + 4 < len(gaussian_maps):  # Ensure surround levels exist
                now_size = gaussian_maps[s].shape
                now_size = (now_size[1], now_size[0])  # (width, height)

                # s vs s+3
                if s + 3 < len(gaussian_maps):
                    tmp = cv2.resize(
                        gaussian_maps[s + 3], now_size, interpolation=cv2.INTER_LINEAR
                    )
                    now_dst = cv2.absdiff(gaussian_maps[s], tmp)
                    dst.append(now_dst)

                # s vs s+4
                if s + 4 < len(gaussian_maps):
                    tmp = cv2.resize(
                        gaussian_maps[s + 4], now_size, interpolation=cv2.INTER_LINEAR
                    )
                    now_dst = cv2.absdiff(gaussian_maps[s], tmp)
                    dst.append(now_dst)

        return dst

    def _gaussian_pyr_CSD(self, src):
        """Create Gaussian pyramid and compute center-surround differences"""
        gaussian_maps = self._create_gaussian_pyramid(src)
        dst = self._center_surround_diff(gaussian_maps)
        return dst

    def _get_intensity_FM(self, I):
        """Get intensity feature maps"""
        return self._gaussian_pyr_CSD(I)

    def _get_color_FM(self, R, G, B):
        """Get color feature maps"""
        # max(R,G,B)
        tmp1 = cv2.max(R, G)
        rgb_max = cv2.max(B, tmp1)
        rgb_max[rgb_max <= 0] = 0.0001  # prevent dividing by 0

        # min(R,G)
        rg_min = cv2.min(R, G)

        # RG = (R-G)/max(R,G,B)
        RG = (R - G) / rgb_max
        # BY = (B-min(R,G)/max(R,G,B)
        BY = (B - rg_min) / rgb_max

        # clamp negative values to 0
        RG[RG < 0] = 0
        BY[BY < 0] = 0

        # obtain feature maps
        RG_FM = self._gaussian_pyr_CSD(RG)
        BY_FM = self._gaussian_pyr_CSD(BY)

        return RG_FM, BY_FM

    def _get_orientation_FM(self, src):
        """Get orientation feature maps"""
        # Create Gaussian pyramid
        gaussian_I = self._create_gaussian_pyramid(src)

        # Convolve with Gabor filters
        gabor_output_0 = [np.empty((1, 1)), np.empty((1, 1))]  # dummy data
        gabor_output_45 = [np.empty((1, 1)), np.empty((1, 1))]
        gabor_output_90 = [np.empty((1, 1)), np.empty((1, 1))]
        gabor_output_135 = [np.empty((1, 1)), np.empty((1, 1))]

        # Apply Gabor filters to pyramid levels 2 through pyramid_levels-1
        max_level = min(self.pyramid_levels - 1, len(gaussian_I) - 1)
        for j in range(2, max_level + 1):
            if j < len(gaussian_I):
                gabor_output_0.append(
                    cv2.filter2D(gaussian_I[j], cv2.CV_32F, self.GaborKernel0)
                )
                gabor_output_45.append(
                    cv2.filter2D(gaussian_I[j], cv2.CV_32F, self.GaborKernel45)
                )
                gabor_output_90.append(
                    cv2.filter2D(gaussian_I[j], cv2.CV_32F, self.GaborKernel90)
                )
                gabor_output_135.append(
                    cv2.filter2D(gaussian_I[j], cv2.CV_32F, self.GaborKernel135)
                )

        # Calculate center-surround differences for each orientation
        CSD_0 = self._center_surround_diff(gabor_output_0)
        CSD_45 = self._center_surround_diff(gabor_output_45)
        CSD_90 = self._center_surround_diff(gabor_output_90)
        CSD_135 = self._center_surround_diff(gabor_output_135)

        # Concatenate
        dst = list(CSD_0)
        dst.extend(CSD_45)
        dst.extend(CSD_90)
        dst.extend(CSD_135)

        return dst

    def _range_normalize(self, src):
        """Standard range normalization"""
        min_val, max_val, _, _ = cv2.minMaxLoc(src)
        if max_val != min_val:
            dst = src / (max_val - min_val) + min_val / (min_val - max_val)
        else:
            dst = src - min_val
        return dst

    def _avg_local_max(self, src):
        """Compute average of local maxima"""
        stepsize = self.default_step_local
        width = src.shape[1]
        height = src.shape[0]

        num_local = 0
        lmax_mean = 0

        for y in range(0, height - stepsize, stepsize):
            for x in range(0, width - stepsize, stepsize):
                local_img = src[y : y + stepsize, x : x + stepsize]
                _, lmax, _, _ = cv2.minMaxLoc(local_img)
                lmax_mean += lmax
                num_local += 1

        return lmax_mean / num_local if num_local > 0 else 0

    def _SM_normalization(self, src):
        """Normalization specific for saliency map model"""
        dst = self._range_normalize(src)
        lmax_mean = self._avg_local_max(dst)
        norm_coeff = (1 - lmax_mean) * (1 - lmax_mean)
        return dst * norm_coeff

    def _normalize_feature_maps(self, FM, target_width, target_height):
        """Normalize feature maps"""
        NFM = []
        # Handle variable number of feature maps based on pyramid levels
        num_maps = len(FM)
        for i in range(num_maps):
            normalized_image = self._SM_normalization(FM[i])
            now_nfm = cv2.resize(
                normalized_image,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            NFM.append(now_nfm)
        return NFM

    def _get_intensity_CM(self, IFM, target_width, target_height):
        """Get intensity conspicuity map"""
        NIFM = self._normalize_feature_maps(IFM, target_width, target_height)
        ICM = sum(NIFM) if NIFM else np.zeros((target_height, target_width))
        return ICM

    def _get_color_CM(self, CFM_RG, CFM_BY, target_width, target_height):
        """Get color conspicuity map"""
        CCM_RG = self._get_intensity_CM(CFM_RG, target_width, target_height)
        CCM_BY = self._get_intensity_CM(CFM_BY, target_width, target_height)
        CCM = CCM_RG + CCM_BY
        return CCM

    def _get_orientation_CM(self, OFM, target_width, target_height):
        """Get orientation conspicuity map"""
        OCM = np.zeros((target_height, target_width))

        # Handle variable number of orientation feature maps
        maps_per_orientation = len(OFM) // 4 if len(OFM) >= 4 else 0

        for i in range(4):
            start_idx = i * maps_per_orientation
            end_idx = start_idx + maps_per_orientation

            if start_idx < len(OFM) and end_idx <= len(OFM):
                # Extract maps for angle = i*45 degrees
                now_ofm = OFM[start_idx:end_idx]
                if now_ofm:  # Only process if we have maps
                    # Get conspicuity map for this angle
                    NOFM = self._get_intensity_CM(now_ofm, target_width, target_height)
                    # Normalize
                    NOFM2 = self._SM_normalization(NOFM)
                    # Accumulate
                    OCM += NOFM2

        return OCM

    # def compute_saliency_map(self, obs):
    def __call__(self, rgba: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Compute Itti-Koch saliency map for the given observation.

        Args:
            obs: Observation dictionary containing 'rgba' key with image data

        Returns:
            sal: Saliency map normalized to [0, 1]
        """
        src = (rgba[:, :, :3] / 255.0).astype(np.uint8)
        
        # Get image dimensions
        size = src.shape
        width = size[1]
        height = size[0]

        # Set target dimensions if not specified
        if self.width is None:
            self.width = width
        if self.height is None:
            self.height = height

        # Extract color channels
        R, G, B, I = self._extract_RGBI(src)

        # Extract feature maps
        IFM = self._get_intensity_FM(I)
        CFM_RG, CFM_BY = self._get_color_FM(R, G, B)
        OFM = self._get_orientation_FM(I)

        # Extract conspicuity maps
        ICM = self._get_intensity_CM(IFM, self.width, self.height)
        CCM = self._get_color_CM(CFM_RG, CFM_BY, self.width, self.height)
        OCM = self._get_orientation_CM(OFM, self.width, self.height)

        # Combine conspicuity maps
        SM_mat = (
            self.weight_intensity * ICM
            + self.weight_color * CCM
            + self.weight_orientation * OCM
        )

        # Normalize
        normalized_SM = self._range_normalize(SM_mat)
        normalized_SM2 = normalized_SM.astype(np.float32)

        # Apply bilateral filter for smoothing
        smoothed_SM = cv2.bilateralFilter(normalized_SM2, 7, 3, 1.55)

        # Resize to original dimensions
        self.SM = cv2.resize(
            smoothed_SM, (width, height), interpolation=cv2.INTER_NEAREST
        )

        # Normalize to [0, 1] range
        if np.max(self.SM) > 0:
            sal = self.SM / np.max(self.SM)
        else:
            sal = self.SM
        return sal.astype(np.float32)


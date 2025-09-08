import cv2
import numpy as np


class IttiKochSaliency:
    """
    Implementation of the Itti-Koch visual attention model.
    Based on "A model of saliency-based visual attention for rapid scene analysis"
    by L. Itti, C. Koch, and E. Niebur, IEEE TPAMI 1998.
    
    This model computes saliency using intensity, color, and orientation conspicuity maps
    through center-surround operations on multi-scale image pyramids.
    """
    
    def __init__(self, width=None, height=None, 
                 weight_intensity=0.33, weight_color=0.33, weight_orientation=0.33,
                 default_step_local=16):
        self.width = width
        self.height = height
        self.weight_intensity = weight_intensity
        self.weight_color = weight_color
        self.weight_orientation = weight_orientation
        self.default_step_local = default_step_local
        self.prev_frame = None
        self.SM = None
        
        # Gabor kernels for orientation detection
        self.GaborKernel0 = np.array([
            [ 1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06 ],
            [ 2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05 ],
            [ 0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076 ],
            [ 0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940 ],
            [ 0.000921261, 0.006375831, -0.174308068, -0.067914552, 1.000000000, -0.067914552, -0.174308068, 0.006375831, 0.000921261 ],
            [ 0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940 ],
            [ 0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076 ],
            [ 2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05 ],
            [ 1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06 ]
        ], dtype=np.float32)
        
        self.GaborKernel45 = np.array([
            [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05,  0.000744712,  0.000132863, -9.04408E-06, -1.01551E-06 ],
            [  2.25320E-05,  0.000925120,  0.002373205, -0.013561362, -0.022947700,  0.000389916,  0.003516954,  0.000288732, -9.04408E-06 ],
            [ -0.000279806,  0.002373205,  0.044837725,  0.052928748, -0.139178011, -0.108372072,  0.000847346,  0.003516954,  0.000132863 ],
            [ -0.001028923, -0.013561362,  0.052928748,  0.460162150,  0.249959607, -0.302454279, -0.108372072,  0.000389916,  0.000744712 ],
            [  3.79931E-05, -0.022947700, -0.139178011,  0.249959607,  1.000000000,  0.249959607, -0.139178011, -0.022947700,  3.79931E-05 ],
            [  0.000744712,  0.003899160, -0.108372072, -0.302454279,  0.249959607,  0.460162150,  0.052928748, -0.013561362, -0.001028923 ],
            [  0.000132863,  0.003516954,  0.000847346, -0.108372072, -0.139178011,  0.052928748,  0.044837725,  0.002373205, -0.000279806 ],
            [ -9.04408E-06,  0.000288732,  0.003516954,  0.000389916, -0.022947700, -0.013561362,  0.002373205,  0.000925120,  2.25320E-05 ],
            [ -1.01551E-06, -9.04408E-06,  0.000132863,  0.000744712,  3.79931E-05, -0.001028923, -0.000279806,  2.25320E-05,  4.04180E-06 ]
        ], dtype=np.float32)
        
        self.GaborKernel90 = np.array([
            [  1.85212E-06,  2.80209E-05,  0.000195076,  0.000624940,  0.000921261,  0.000624940,  0.000195076,  2.80209E-05,  1.85212E-06 ],
            [  1.28181E-05,  0.000193926,  0.001350077,  0.004325061,  0.006375831,  0.004325061,  0.001350077,  0.000193926,  1.28181E-05 ],
            [ -0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433 ],
            [ -0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537 ],
            [  0.002010422,  0.030415784,  0.211749204,  0.678352526,  1.000000000,  0.678352526,  0.211749204,  0.030415784,  0.002010422 ],
            [ -0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537 ],
            [ -0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433 ],
            [  1.28181E-05,  0.000193926,  0.001350077,  0.004325061,  0.006375831,  0.004325061,  0.001350077,  0.000193926,  1.28181E-05 ],
            [  1.85212E-06,  2.80209E-05,  0.000195076,  0.000624940,  0.000921261,  0.000624940,  0.000195076,  2.80209E-05,  1.85212E-06 ]
        ], dtype=np.float32)
        
        self.GaborKernel135 = np.array([
            [ -1.01551E-06, -9.04408E-06,  0.000132863,  0.000744712,  3.79931E-05, -0.001028923, -0.000279806, 2.2532E-05, 4.0418E-06 ],
            [ -9.04408E-06,  0.000288732,  0.003516954,  0.000389916, -0.022947700, -0.013561362, 0.002373205, 0.00092512, 2.2532E-05 ],
            [  0.000132863,  0.003516954,  0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806 ],
            [  0.000744712,  0.000389916, -0.108372072, -0.302454279,  0.249959607, 0.46016215, 0.052928748, -0.013561362, -0.001028923 ],
            [  3.79931E-05, -0.022947700, -0.139178011,  0.249959607,  1.000000000, 0.249959607, -0.139178011, -0.0229477, 3.79931E-05 ],
            [ -0.001028923, -0.013561362,  0.052928748,  0.460162150,  0.249959607, -0.302454279, -0.108372072, 0.000389916, 0.000744712 ],
            [ -0.000279806,  0.002373205,  0.044837725,  0.052928748, -0.139178011, -0.108372072, 0.000847346, 0.003516954, 0.000132863 ],
            [  2.25320E-05,  0.000925120,  0.002373205, -0.013561362, -0.022947700, 0.000389916, 0.003516954, 0.000288732, -9.04408E-06 ],
            [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05 , 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06 ]
        ], dtype=np.float32)

    def _extract_RGBI(self, input_image):
        """Extract R, G, B, and intensity channels"""
        src = np.float32(input_image) * 1./255
        (B, G, R) = cv2.split(src)
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return R, G, B, I

    def _create_gaussian_pyramid(self, src):
        """Create 9-level Gaussian pyramid"""
        dst = [src]
        for i in range(1, 9):
            now_dst = cv2.pyrDown(dst[i-1])
            dst.append(now_dst)
        return dst

    def _center_surround_diff(self, gaussian_maps):
        """Compute center-surround differences"""
        dst = []
        for s in range(2, 5):
            now_size = gaussian_maps[s].shape
            now_size = (now_size[1], now_size[0])  # (width, height)
            
            # s vs s+3
            tmp = cv2.resize(gaussian_maps[s+3], now_size, interpolation=cv2.INTER_LINEAR)
            now_dst = cv2.absdiff(gaussian_maps[s], tmp)
            dst.append(now_dst)
            
            # s vs s+4
            tmp = cv2.resize(gaussian_maps[s+4], now_size, interpolation=cv2.INTER_LINEAR)
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
        gabor_output_0 = [np.empty((1,1)), np.empty((1,1))]  # dummy data
        gabor_output_45 = [np.empty((1,1)), np.empty((1,1))]
        gabor_output_90 = [np.empty((1,1)), np.empty((1,1))]
        gabor_output_135 = [np.empty((1,1)), np.empty((1,1))]
        
        for j in range(2, 9):
            gabor_output_0.append(cv2.filter2D(gaussian_I[j], cv2.CV_32F, self.GaborKernel0))
            gabor_output_45.append(cv2.filter2D(gaussian_I[j], cv2.CV_32F, self.GaborKernel45))
            gabor_output_90.append(cv2.filter2D(gaussian_I[j], cv2.CV_32F, self.GaborKernel90))
            gabor_output_135.append(cv2.filter2D(gaussian_I[j], cv2.CV_32F, self.GaborKernel135))
        
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
                local_img = src[y:y+stepsize, x:x+stepsize]
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
        for i in range(6):
            normalized_image = self._SM_normalization(FM[i])
            now_nfm = cv2.resize(normalized_image, (target_width, target_height), 
                               interpolation=cv2.INTER_LINEAR)
            NFM.append(now_nfm)
        return NFM

    def _get_intensity_CM(self, IFM, target_width, target_height):
        """Get intensity conspicuity map"""
        NIFM = self._normalize_feature_maps(IFM, target_width, target_height)
        ICM = sum(NIFM)
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
        
        for i in range(4):
            # Extract maps for angle = i*45 degrees
            now_ofm = OFM[i*6:(i+1)*6]
            # Get conspicuity map for this angle
            NOFM = self._get_intensity_CM(now_ofm, target_width, target_height)
            # Normalize
            NOFM2 = self._SM_normalization(NOFM)
            # Accumulate
            OCM += NOFM2
            
        return OCM

    def compute_saliency(self, src):
        """
        Compute Itti-Koch saliency map for the given image.
        
        Args:
            src: Input image as numpy array (BGR format)
            
        Returns:
            sal: Saliency map normalized to [0, 255]
        """
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
        SM_mat = (self.weight_intensity * ICM + 
                  self.weight_color * CCM + 
                  self.weight_orientation * OCM)

        # Normalize
        normalized_SM = self._range_normalize(SM_mat)
        normalized_SM2 = normalized_SM.astype(np.float32)
        
        # Apply bilateral filter for smoothing
        smoothed_SM = cv2.bilateralFilter(normalized_SM2, 7, 3, 1.55)
        
        # Resize to original dimensions
        self.SM = cv2.resize(smoothed_SM, (width, height), interpolation=cv2.INTER_NEAREST)

        # Convert to [0, 255] range
        sal = (self.SM * 255).astype(np.uint8)
        return sal

    def compute_saliency_from_path(self, img_path):
        """
        Compute saliency map from image file path.
        
        Args:
            img_path: Path to input image file
            
        Returns:
            sal: Saliency map normalized to [0, 255]
        """
        img = cv2.imread(img_path)
        return self.compute_saliency(img)
# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from scipy import ndimage as ndi

from .strategies import SalienceStrategy


def sobel_filter(img: np.ndarray):
    """Apply Sobel filter to a grayscale image (2D array).

    Returns gradients in x, y, and the magnitude.
    """
    # Define Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Convolve image with kernels
    gx = ndimage.convolve(img.astype(np.float32), Kx)
    gy = ndimage.convolve(img.astype(np.float32), Ky)

    # Edge magnitude
    mag = np.hypot(gx, gy)  # sqrt(gx**2 + gy**2)
    mag = mag / (mag.max() + 1e-8)  # normalize [0,1]

    return gx, gy, mag


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image [H,W,3] to grayscale [H,W].
    Input can be uint8 (0–255) or float (0–1).
    """
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0  # normalize

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def color_opponency_map(rgb, target):
    """Emphasize pixels matching a target color.

    Parameters
    ----------
    rgb : np.ndarray
        [H,W,3] float32 image in [0,1].
    target : tuple or list of 3
        Target color (R,G,B), e.g. (1,0,0) for red.

    Returns
    -------
    map : np.ndarray
        [H,W] float32 map in [0,1].
    """

    # target = np.array(target, dtype=np.float32)
    # target /= np.linalg.norm(target) + 1e-8  # normalize

    # dot product per pixel
    dot = np.tensordot(rgb, target, axes=([2], [0]))

    # rescale from [-1,1] → [0,1]
    map_ = (dot + 1) / 2.0
    return map_.astype(np.float32)


# ---------- helpers ----------


def norm01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def halfwave(x):
    # simple rectification like neuronal ON channels
    return np.maximum(x, 0.0).astype(np.float32)


def dog_center_surround(img, sigma_c, sigma_s, k=1.0):
    # Difference-of-Gaussians (center-surround)
    c = ndimage.gaussian_filter(img, sigma=sigma_c, mode="reflect")
    s = ndimage.gaussian_filter(img, sigma=sigma_s, mode="reflect")
    return k * (c - s).astype(np.float32)


def multiscale_cs(img, scales: Tuple[Tuple[float, float], ...]):
    # sum of rectified center-surround across scales
    acc = np.zeros(img.shape[:2], dtype=np.float32)
    for sc, ss in scales:
        m = halfwave(dog_center_surround(img, sc, ss))
        acc += norm01(m)
    return norm01(acc)


# ---------- color & luminance feature maps ----------


def rgb_to_gray(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def opponency_RG(rgb):
    # classic L-M (red-green) opponent; rectified into ON channels
    r, g = rgb[..., 0], rgb[..., 1]
    rg_on = halfwave(r - g)
    gr_on = halfwave(g - r)
    return rg_on, gr_on  # (red-on, green-on)


def opponency_BY(rgb):
    # S - (L+M) ~ blue - yellow (yellow ≈ (R+G)/2)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.5 * (r + g)
    b_on = halfwave(b - y)
    y_on = halfwave(y - b)
    return b_on, y_on  # (blue-on, yellow-on)


def color_salience(rgb):
    # center-surround on opponent channels + across-scale normalization
    scales = ((1.0, 3.0), (2.0, 6.0), (4.0, 12.0))
    rg_on, gr_on = opponency_RG(rgb)
    b_on, y_on = opponency_BY(rgb)

    RG = multiscale_cs(rg_on, scales) + multiscale_cs(gr_on, scales)
    BY = multiscale_cs(b_on, scales) + multiscale_cs(y_on, scales)
    return norm01(RG), norm01(BY)


def luminance_salience(rgb):
    lum = rgb_to_gray(rgb)
    scales = ((1.0, 3.0), (2.0, 6.0), (4.0, 12.0))
    return multiscale_cs(lum, scales)


# ---------- orientation / texture feature maps ----------


def sobel_xy(gray):
    # 3x3 Sobel via ndimage
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    gx = ndimage.convolve(gray, Kx, mode="reflect")
    gy = ndimage.convolve(gray, Ky, mode="reflect")
    return gx, gy


def orientation_energy(gray, num_bins: int = 4):
    """
    Simple orientation channels using gradient steering into bins.
    Returns a list of maps (length num_bins) plus an overall magnitude.
    """
    gx, gy = sobel_xy(gray)
    mag = np.hypot(gx, gy).astype(np.float32)
    ang = np.arctan2(gy, gx) + np.pi  # [0, 2pi)

    bins = []
    for k in range(num_bins):
        theta = k * np.pi / num_bins  # 0..pi
        # soft binning with cosine tuning (like simple cell)
        resp = np.cos(ang - theta)
        resp = halfwave(resp) * mag
        bins.append(norm01(resp))
    return bins, norm01(mag)


def oriented_salience(rgb, num_bins=4):
    gray = rgb_to_gray(rgb)
    bins, mag = orientation_energy(gray, num_bins=num_bins)
    # center-surround on each orientation channel
    scales = ((1.0, 3.0), (2.0, 6.0))
    acc = np.zeros_like(gray, dtype=np.float32)
    for ch in bins:
        acc += multiscale_cs(ch, scales)
    # mix with overall edge magnitude for sharper contours
    return norm01(0.7 * acc + 0.3 * mag)


# ---------- depth feature maps ----------


def depth_nearness(depth, robust=True):
    """
    Convert depth (meters; 0 or NaN = invalid) to nearness in [0,1],
    emphasizing nearer objects.
    """
    d = depth.astype(np.float32).copy()
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    if robust:
        med = np.nanmedian(d)
        mad = np.nanmedian(np.abs(d - med)) + 1e-6
        z = (med - d) / (1.4826 * mad + 1e-6)  # nearer -> larger positive
    else:
        z = -d
    z = np.nan_to_num(z, nan=0.0)
    return norm01(z)


def depth_confidence(depth):
    """
    Cheap confidence: valid pixels with low local variance.
    """
    d = depth.astype(np.float32)
    valid = (np.isfinite(d) & (d > 0)).astype(np.float32)
    sm = ndimage.gaussian_filter(np.nan_to_num(d, nan=0.0), 1.0, mode="reflect")
    var = ndimage.gaussian_filter(
        (np.nan_to_num(d, nan=0.0) - sm) ** 2, 2.0, mode="reflect"
    )
    var = norm01(var)
    conf = valid * (1.0 - var)
    return conf.astype(np.float32)


def depth_salience(depth):
    """
    Near-field pop-out + center-surround + depth edges.
    """
    near = depth_nearness(depth)  # nearer → higher
    # center-surround on nearness
    scales = ((1.0, 3.0), (2.0, 6.0), (4.0, 12.0))
    cs = multiscale_cs(near, scales)
    # add depth edges to hug object boundaries
    gx, gy = sobel_xy(near)
    edges = norm01(np.hypot(gx, gy))
    ds = norm01(0.8 * cs + 0.2 * edges)
    return ds


# ---------- fusion & post ----------


def fuse_maps(maps: Dict[str, np.ndarray], weights: Dict[str, float]):
    """
    Weighted geometric mean (robust to differing ranges, rewards consensus).
    maps: dict of {name: [H,W] map in [0,1]}
    weights: dict of {name: exponent weight}
    """
    # avoid log(0)
    eps = 1e-6
    log_acc = None
    for k, m in maps.items():
        w = float(weights.get(k, 1.0))
        m = np.clip(m, eps, 1.0).astype(np.float32)
        term = w * np.log(m)
        log_acc = term if log_acc is None else (log_acc + term)
    fused = np.exp(log_acc)
    return norm01(fused)


def spectral_residual_saliency(gray):
    # gray: [H,W] float32 in [0,1]
    f = np.fft.fft2(gray)
    amp = np.abs(f)
    log_amp = np.log(amp + 1e-8)
    # local average in log-spectrum (box or gaussian)
    log_avg = ndimage.uniform_filter(log_amp, size=3, mode="reflect")
    resid = log_amp - log_avg
    # reconstruct with original phase
    f_resid = np.exp(resid + 1j * np.angle(f))
    sal = np.abs(np.fft.ifft2(f_resid)) ** 2
    sal = ndimage.gaussian_filter(sal, 3.0, mode="reflect")
    return norm01(sal.astype(np.float32))


def fuse_maps_mean(maps, weights):
    num = 0.0
    den = 0.0
    for k, m in maps.items():
        w = float(weights.get(k, 1.0))
        num += w * m
        den += w
    S = num / (den + 1e-8)
    # mild contrast boost (acts like a soft sigmoid)
    return norm01(S**1.2)


from typing import Dict, List, Optional

import numpy as np


def _norm01(x, eps=1e-8):
    mn, mx = x.min(), x.max()
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _maps_to_matrix(maps: Dict[str, np.ndarray], keys: Optional[List[str]] = None):
    """Stack maps -> X shape [N, K] with per-map z-scoring."""
    if keys is None:
        keys = list(maps.keys())
    H, W = next(iter(maps.values())).shape
    X = []
    for k in keys:
        m = maps[k].astype(np.float32).reshape(-1)
        # z-score per map to compare structure not scale
        mu = m.mean()
        sd = m.std() + 1e-8
        X.append((m - mu) / sd)
    X = np.stack(X, axis=1)  # [N,K]
    return X, (H, W), keys


def corr_shrink_weights(
    maps: Dict[str, np.ndarray],
    base_weights: Dict[str, float],
    keys: Optional[List[str]] = None,
    power: float = 1.0,
):
    """
    Compute redundancy-aware weights: w_i' = w_i / (1 + sum_j |corr(i,j)|^{power}), j≠i
    Returns dict of shrunken, normalized weights.
    """
    X, _, keys = _maps_to_matrix(maps, keys)
    K = X.shape[1]
    # correlation matrix (abs)
    C = np.corrcoef(X, rowvar=False)  # [K,K]
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    R = np.sum(np.abs(C) ** power, axis=1)  # redundancy score per map
    # shrink base weights by redundancy
    w = np.array([float(base_weights.get(k, 1.0)) for k in keys], dtype=np.float32)
    w_shrunk = w / (1.0 + R)
    w_shrunk = np.clip(w_shrunk, 0.0, None)
    # normalize to keep total weight comparable
    if w_shrunk.sum() > 0:
        w_shrunk /= w_shrunk.sum()
    return {k: float(ws) for k, ws in zip(keys, w_shrunk)}, C


def fuse_mean_redundancy_aware(
    maps: Dict[str, np.ndarray],
    base_weights: Dict[str, float],
    keys: Optional[List[str]] = None,
    power: float = 1.0,
):
    """
    Arithmetic mean fusion with correlation-aware weight shrinking.
    """
    if keys is None:
        keys = list(maps.keys())
    w_shrunk, C = corr_shrink_weights(maps, base_weights, keys, power=power)
    # weighted mean
    num = 0.0
    den = 0.0
    for k in keys:
        wk = float(w_shrunk.get(k, 0.0))
        num = num + wk * maps[k].astype(np.float32)
        den = den + wk
    S = num / (den + 1e-8)
    return _norm01(S), w_shrunk, C


# ---------- end-to-end pipeline ----------


def salience_rgbd(rgb, depth=None, weight_scheme="indoor_default"):
    # --- features as before ---
    L = luminance_salience(rgb)
    RG, BY = color_salience(rgb)
    OR = oriented_salience(rgb)

    # new: spectral residual on luminance
    SR = spectral_residual_saliency(rgb_to_gray(rgb))

    maps = {"L": L, "RG": RG, "BY": BY, "OR": OR, "SR": SR}

    if depth is not None:
        D = depth_salience(depth)
        C = depth_confidence(depth)
        maps["D"] = norm01(D * (0.3 + 0.7 * C))

        valid = np.isfinite(depth) & (depth > 0)
        if np.any(valid):
            med = np.median(depth[valid])
            dmin = np.min(depth[valid])
            if (med - dmin) > 0.5:  # meters
                maps["D"] = norm01(maps["D"] ** 1.2)

    # weights
    if weight_scheme == "indoor_default":
        weights = {"L": 0.6, "RG": 0.8, "BY": 0.8, "OR": 0.9, "SR": 1.2}
        if depth is not None:
            weights["D"] = 0.8
    else:
        weights = {k: 1.0 for k in maps.keys()}

    # use mean fusion instead of geometric
    S = fuse_maps_mean(maps, weights)
    # S, _, _ = fuse_mean_redundancy_aware(maps, weights)
    # S = ndimage.gaussian_filter(S, 0.1, mode="reflect")  # a touch of smoothing
    maps["S"] = norm01(S)

    return maps

# TODO: Remove this function
def compute_saliency(obs: dict):
    rgb = obs["rgba"][:, :, :3] / 255.0
    depth = obs["depth"]
    maps = salience_rgbd(rgb, depth)
    return maps["S"]


class BioSalienceStrategy(SalienceStrategy):

    def __call__(self, rgba: np.ndarray, depth: np.ndarray) -> np.ndarray:
        rgb = rgba[:, :, :3] / 255.0
        maps = salience_rgbd(rgb, depth)
        return maps["S"]

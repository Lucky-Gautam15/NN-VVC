"""
Evaluation metrics module for NN-VVC.

Provides robust calculation of MSE, PSNR (single component, RGB, YUV420 weighted),
Multi-Scale SSIM (MS-SSIM), bitrate (bps / kbps), and bits-per-pixel (bpp).

Paper Reference:
    "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
     image coding for machines", Section V (Experimental Results).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.vvc.muxer import NNVVCDeMuxer


@dataclass
class RDPoint:
    """Represents a single Rate-Distortion operating point."""
    qp: int
    bitrate_kbps: float
    bpp: float
    total_bits: int
    psnr_y: Optional[float] = None
    psnr_u: Optional[float] = None
    psnr_v: Optional[float] = None
    psnr_yuv: Optional[float] = None
    psnr_rgb: Optional[float] = None
    ms_ssim: Optional[float] = None
    extra_metrics: Optional[Dict[str, float]] = None


@dataclass
class SequenceMetrics:
    """Aggregated evaluation metrics for a decoded video sequence."""
    sequence_name: str
    width: int
    height: int
    frame_count: int
    framerate: float
    rd_points: List[RDPoint]


# -----------------------------------------------------------------------------
# MSE and PSNR Calculation
# -----------------------------------------------------------------------------

def _to_numpy_float64(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert input NumPy array or PyTorch tensor to float64 NumPy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float64)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float64)
    else:
        raise TypeError(f"Unsupported array type {type(data)}. Expected np.ndarray or torch.Tensor.")


def _infer_data_range(data: np.ndarray, data_range: Optional[float] = None) -> float:
    """Infer data range based on dtype if not explicitly provided."""
    if data_range is not None:
        if data_range <= 0:
            raise ValueError(f"data_range must be strictly positive, got {data_range}")
        return float(data_range)
    # If integer-like max value > 1, assume 255.0, otherwise 1.0
    if np.max(data) > 1.0 or data.dtype in (np.uint8, np.int16, np.uint16, np.int32, np.int64):
        return 255.0
    return 1.0


def calculate_mse(
    reference: Union[np.ndarray, torch.Tensor],
    reconstruction: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Calculate Mean Squared Error (MSE) between reference and reconstruction.

    Args:
        reference: Ground-truth reference image/tensor.
        reconstruction: Reconstructed image/tensor.

    Returns:
        Mean squared error as a float >= 0.0.
    """
    ref_arr = _to_numpy_float64(reference)
    rec_arr = _to_numpy_float64(reconstruction)

    if ref_arr.shape != rec_arr.shape:
        raise ValueError(
            f"Shape mismatch in calculate_mse: reference {ref_arr.shape} vs reconstruction {rec_arr.shape}."
        )

    if ref_arr.size == 0:
        raise ValueError("Cannot calculate MSE on empty arrays.")

    return float(np.mean((ref_arr - rec_arr) ** 2))


def calculate_psnr(
    reference: Union[np.ndarray, torch.Tensor],
    reconstruction: Union[np.ndarray, torch.Tensor],
    data_range: Optional[float] = None,
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        reference: Ground-truth reference.
        reconstruction: Reconstructed image.
        data_range: Peak value of the signal (e.g. 1.0 for float [0,1], 255.0 for 8-bit, 1023.0 for 10-bit).
                    If None, inferred automatically.

    Returns:
        PSNR in dB (float('inf') if MSE == 0).
    """
    ref_arr = _to_numpy_float64(reference)
    rec_arr = _to_numpy_float64(reconstruction)

    dr = _infer_data_range(ref_arr, data_range)
    mse = calculate_mse(ref_arr, rec_arr)

    if mse == 0.0:
        return float("inf")

    return float(10.0 * math.log10((dr ** 2) / mse))


def calculate_psnr_yuv(
    reference: Tuple[np.ndarray, np.ndarray, np.ndarray],
    reconstruction: Tuple[np.ndarray, np.ndarray, np.ndarray],
    bit_depth: int = 8,
    weights: Tuple[float, float, float] = (6.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Calculate per-component (Y, U, V) and standard weighted YUV PSNR for planar YUV420.

    Standard JVET YUV-PSNR weighting convention:
        PSNR_YUV = (6 * PSNR_Y + PSNR_U + PSNR_V) / 8

    Args:
        reference: Tuple of (Y, U, V) planar arrays for reference.
        reconstruction: Tuple of (Y, U, V) planar arrays for reconstruction.
        bit_depth: Video bit depth (8, 10, etc.).
        weights: Weights for (Y, U, V) in aggregate score (default 6:1:1).

    Returns:
        Dictionary with keys 'psnr_y', 'psnr_u', 'psnr_v', 'psnr_yuv', 'mse_y', 'mse_u', 'mse_v'.
    """
    if len(reference) != 3 or len(reconstruction) != 3:
        raise ValueError("Reference and reconstruction must each be a 3-tuple (Y, U, V).")

    y_ref, u_ref, v_ref = reference
    y_rec, u_rec, v_rec = reconstruction

    max_val = float((1 << bit_depth) - 1)

    mse_y = calculate_mse(y_ref, y_rec)
    mse_u = calculate_mse(u_ref, u_rec)
    mse_v = calculate_mse(v_ref, v_rec)

    psnr_y = calculate_psnr(y_ref, y_rec, data_range=max_val)
    psnr_u = calculate_psnr(u_ref, u_rec, data_range=max_val)
    psnr_v = calculate_psnr(v_ref, v_rec, data_range=max_val)

    w_y, w_u, w_v = weights
    total_w = w_y + w_u + w_v
    if total_w <= 0:
        raise ValueError(f"Sum of weights must be positive, got {total_w}")

    # Aggregate weighted PSNR
    if any(math.isinf(p) for p in (psnr_y, psnr_u, psnr_v)):
        psnr_yuv = float("inf") if all(math.isinf(p) for p in (psnr_y, psnr_u, psnr_v)) else (
            w_y * (psnr_y if not math.isinf(psnr_y) else 100.0)
            + w_u * (psnr_u if not math.isinf(psnr_u) else 100.0)
            + w_v * (psnr_v if not math.isinf(psnr_v) else 100.0)
        ) / total_w
    else:
        psnr_yuv = (w_y * psnr_y + w_u * psnr_u + w_v * psnr_v) / total_w

    return {
        "psnr_y": psnr_y,
        "psnr_u": psnr_u,
        "psnr_v": psnr_v,
        "psnr_yuv": psnr_yuv,
        "mse_y": mse_y,
        "mse_u": mse_u,
        "mse_v": mse_v,
    }


# -----------------------------------------------------------------------------
# Multi-Scale SSIM (MS-SSIM)
# -----------------------------------------------------------------------------

def _fspecial_gaussian_1d(size: int, sigma: float) -> torch.Tensor:
    """Generate 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_filter_2d(img: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    """Apply separable 2D Gaussian filter to a 4D tensor (B, C, H, W)."""
    b, c, h, w = img.shape
    k = kernel_1d.to(img.device, img.dtype)
    k_h = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    k_w = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    pad = k.shape[0] // 2

    # Convolve height then width
    out = F.conv2d(img, k_h, padding=(pad, 0), groups=c)
    out = F.conv2d(out, k_w, padding=(0, pad), groups=c)
    return out


def _ssim_one_scale(
    img1: torch.Tensor,
    img2: torch.Tensor,
    kernel_1d: torch.Tensor,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute single-scale SSIM and contrast-structure measure.

    Returns:
        (cs_map, ssim_map)
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu1 = _gaussian_filter_2d(img1, kernel_1d)
    mu2 = _gaussian_filter_2d(img2, kernel_1d)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_filter_2d(img1 * img1, kernel_1d) - mu1_sq
    sigma2_sq = _gaussian_filter_2d(img2 * img2, kernel_1d) - mu2_sq
    sigma12 = _gaussian_filter_2d(img1 * img2, kernel_1d) - mu1_mu2

    # Avoid negative variance due to numerical precision
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    cs_map = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2.0 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    return cs_map.mean(dim=(-2, -1)), ssim_map.mean(dim=(-2, -1))


def calculate_ms_ssim(
    reference: Union[np.ndarray, torch.Tensor],
    reconstruction: Union[np.ndarray, torch.Tensor],
    data_range: Optional[float] = None,
    weights: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    kernel_size: int = 11,
    kernel_sigma: float = 1.5,
) -> float:
    """
    Calculate Multi-Scale Structural Similarity Index (MS-SSIM) (Wang et al., 2003).

    Args:
        reference: Reference image array (H, W), (H, W, C), or tensor (C, H, W) / (1, C, H, W).
        reconstruction: Reconstructed image array/tensor matching reference.
        data_range: Dynamic range of input signal (default inferred as 1.0 or 255.0).
        weights: Weights for the 5 pyramid levels.
        kernel_size: Gaussian kernel window size (default 11).
        kernel_sigma: Gaussian kernel standard deviation (default 1.5).

    Returns:
        MS-SSIM score between 0.0 and 1.0 (1.0 for identical images).
    """
    def _to_4d_tensor(val: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(val, np.ndarray):
            if val.ndim == 2:
                # (H, W) -> (1, 1, H, W)
                t = torch.from_numpy(val.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            elif val.ndim == 3:
                # (H, W, C) -> (1, C, H, W)
                t = torch.from_numpy(val.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported NumPy array dimensions for MS-SSIM: {val.shape}")
        elif isinstance(val, torch.Tensor):
            t = val.detach().cpu().to(torch.float32)
            if t.ndim == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.ndim == 3:
                t = t.unsqueeze(0)
            elif t.ndim != 4:
                raise ValueError(f"Unsupported PyTorch tensor dimensions for MS-SSIM: {t.shape}")
        else:
            raise TypeError(f"Unsupported input type: {type(val)}")
        return t

    t_ref = _to_4d_tensor(reference)
    t_rec = _to_4d_tensor(reconstruction)

    if t_ref.shape != t_rec.shape:
        raise ValueError(
            f"Shape mismatch in calculate_ms_ssim: reference {t_ref.shape} vs reconstruction {t_rec.shape}."
        )

    _, _, h, w = t_ref.shape
    num_scales = len(weights)
    min_dim = min(h, w)
    min_needed = (kernel_size - 1) * (2 ** (num_scales - 1)) + 1
    if min_dim < min_needed:
        # If image is smaller than required 5 scales, fallback to smaller number of scales or raise
        num_scales = max(1, int(math.log2(min_dim / (kernel_size - 1))))
        scaled_w = weights[:num_scales]
        weights = tuple(w_i / sum(scaled_w) for w_i in scaled_w)

    dr = _infer_data_range(t_ref.numpy(), data_range)
    kernel_1d = _fspecial_gaussian_1d(kernel_size, kernel_sigma)

    mcs_list: List[torch.Tensor] = []
    curr_ref = t_ref
    curr_rec = t_rec

    for i in range(len(weights)):
        cs_map, ssim_map = _ssim_one_scale(curr_ref, curr_rec, kernel_1d, data_range=dr)
        if i < len(weights) - 1:
            mcs_list.append(cs_map)
            # 2x2 average downsampling for next scale
            curr_ref = F.avg_pool2d(curr_ref, kernel_size=2, stride=2)
            curr_rec = F.avg_pool2d(curr_rec, kernel_size=2, stride=2)
        else:
            mcs_list.append(ssim_map)

    # Compute weighted product: prod(mcs_i ** w_i)
    overall_score = torch.ones_like(mcs_list[0])
    for w_i, mcs_i in zip(weights, mcs_list):
        # Clamp mcs_i to avoid negative values before fractional exponent
        clamped_mcs = torch.clamp(mcs_i, min=1e-8, max=1.0)
        overall_score = overall_score * (clamped_mcs ** w_i)

    return float(torch.clamp(overall_score.mean(), 0.0, 1.0).item())


# -----------------------------------------------------------------------------
# Bitrate and Bits-Per-Pixel (BPP) Calculations
# -----------------------------------------------------------------------------

def calculate_bitrate_bits_per_second(
    bitstream_bytes: int,
    frame_count: int,
    fps: float,
) -> float:
    """
    Calculate video transmission bitrate in bits per second (bps).

    Args:
        bitstream_bytes: Total compressed byte count (must be >= 0).
        frame_count: Total coded frames (must be > 0).
        fps: Frames per second (must be > 0).

    Returns:
        Bitrate in bits per second (float).
    """
    if bitstream_bytes < 0:
        raise ValueError(f"bitstream_bytes cannot be negative, got {bitstream_bytes}")
    if frame_count <= 0:
        raise ValueError(f"frame_count must be strictly positive, got {frame_count}")
    if fps <= 0:
        raise ValueError(f"fps must be strictly positive, got {fps}")

    total_bits = bitstream_bytes * 8.0
    duration_sec = frame_count / float(fps)
    return total_bits / duration_sec


def calculate_bpp(
    bitstream_bytes: int,
    width: int,
    height: int,
    frame_count: int,
) -> float:
    """
    Calculate average bits per pixel (bpp) across the video sequence.

    Formula:
        bpp = total_bits / (width * height * frame_count)

    Args:
        bitstream_bytes: Total compressed bytes.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frame_count: Number of frames.

    Returns:
        Bits per pixel as a float >= 0.0.
    """
    if bitstream_bytes < 0:
        raise ValueError(f"bitstream_bytes cannot be negative, got {bitstream_bytes}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions must be positive integers, got {width}x{height}")
    if frame_count <= 0:
        raise ValueError(f"frame_count must be strictly positive, got {frame_count}")

    total_bits = bitstream_bytes * 8.0
    total_pixels = float(width * height * frame_count)
    return total_bits / total_pixels


def extract_nnvvc_payload_breakdown(
    container_path_or_bytes: Union[str, Path, bytes],
) -> Dict[str, Union[int, float]]:
    """
    Extract payload size breakdown (neural bits, VTM bits, total bits, header bits)
    from a .nnvvc file without decompressing the payloads.
    """
    header = NNVVCDeMuxer.read_header(container_path_or_bytes)
    neural_bytes = header.neural_payload_size
    vtm_bytes = header.vtm_payload_size

    if isinstance(container_path_or_bytes, (str, Path)):
        total_file_bytes = Path(container_path_or_bytes).stat().st_size
    else:
        total_file_bytes = len(container_path_or_bytes)

    header_bytes = total_file_bytes - (neural_bytes + vtm_bytes)

    return {
        "neural_bytes": neural_bytes,
        "vtm_bytes": vtm_bytes,
        "header_bytes": header_bytes,
        "total_bytes": total_file_bytes,
        "neural_bits": neural_bytes * 8,
        "vtm_bits": vtm_bytes * 8,
        "header_bits": header_bytes * 8,
        "total_bits": total_file_bytes * 8,
    }

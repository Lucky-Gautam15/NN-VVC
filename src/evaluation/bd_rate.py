"""
Bjøntegaard Delta metrics for NN-VVC evaluation.

Implements BD-Rate and BD-PSNR computation using cubic polynomial fitting
on log-rate vs. distortion curves, following the standard methodology from:

    G. Bjontegaard, "Calculation of average PSNR differences between RD-curves,"
    ITU-T SG16/Q6, VCEG-M33, April 2001.

Sign conventions (standard interpretation):
    BD-Rate:
        Negative = test codec requires FEWER bits for equivalent quality (improvement).
        Positive = test codec requires MORE bits for equivalent quality (degradation).
    BD-PSNR:
        Positive = test codec provides HIGHER quality at equivalent bitrate (improvement).
        Negative = test codec provides LOWER quality at equivalent bitrate (degradation).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class BDResult:
    """Result of a Bjøntegaard Delta comparison between two codecs."""
    bd_rate_percent: float
    bd_psnr_db: float
    anchor_codec: str
    test_codec: str
    num_anchor_points: int
    num_test_points: int
    common_rate_range: Optional[Tuple[float, float]] = None
    common_psnr_range: Optional[Tuple[float, float]] = None


def _validate_rd_points(
    rates: np.ndarray,
    metrics: np.ndarray,
    label: str,
    min_points: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and sort RD points.

    Args:
        rates: Array of bitrates (must be strictly positive).
        metrics: Array of quality metric values (e.g. PSNR).
        label: Human-readable label for error messages.
        min_points: Minimum number of distinct rate points required.

    Returns:
        Sorted (rates, metrics) by ascending rate.

    Raises:
        ValueError on invalid inputs.
    """
    if len(rates) != len(metrics):
        raise ValueError(
            f"{label}: rate array length ({len(rates)}) != metric array length ({len(metrics)})."
        )

    if len(rates) < min_points:
        raise ValueError(
            f"{label}: requires at least {min_points} RD points, got {len(rates)}."
        )

    if np.any(rates <= 0):
        raise ValueError(f"{label}: all rate values must be strictly positive.")

    if np.any(np.isnan(rates)) or np.any(np.isnan(metrics)):
        raise ValueError(f"{label}: rate and metric arrays must not contain NaN values.")

    if np.any(np.isinf(rates)):
        raise ValueError(f"{label}: rate values must be finite.")

    # Sort by ascending rate
    sort_idx = np.argsort(rates)
    sorted_rates = rates[sort_idx]
    sorted_metrics = metrics[sort_idx]

    # Check for duplicate rates
    unique_rates = np.unique(sorted_rates)
    if len(unique_rates) < min_points:
        raise ValueError(
            f"{label}: after removing duplicate rates, only {len(unique_rates)} distinct points "
            f"remain ({min_points} required)."
        )

    # If duplicates exist, average the corresponding metric values
    if len(unique_rates) < len(sorted_rates):
        averaged_metrics = np.array([
            np.mean(sorted_metrics[sorted_rates == r]) for r in unique_rates
        ])
        sorted_rates = unique_rates
        sorted_metrics = averaged_metrics

    return sorted_rates, sorted_metrics


def _polyfit_integral(log_rates: np.ndarray, metrics: np.ndarray, lo: float, hi: float) -> float:
    """
    Fit a 3rd-order polynomial metric = f(log_rate) and integrate over [lo, hi].

    Args:
        log_rates: Log-transformed rate values.
        metrics: Corresponding quality metric values (e.g. PSNR).
        lo: Lower integration bound (in log-rate domain).
        hi: Upper integration bound (in log-rate domain).

    Returns:
        Definite integral of the fitted polynomial over [lo, hi].
    """
    # Fit: metric = a * x^3 + b * x^2 + c * x + d, where x = log(rate)
    coeffs = np.polyfit(log_rates, metrics, deg=3)
    # np.polyfit returns coefficients from highest to lowest degree
    a, b, c, d = coeffs

    # Integrate analytically: int(a*x^3 + b*x^2 + c*x + d) = a/4*x^4 + b/3*x^3 + c/2*x^2 + d*x
    def antiderivative(x: float) -> float:
        return (a / 4.0) * x**4 + (b / 3.0) * x**3 + (c / 2.0) * x**2 + d * x

    return antiderivative(hi) - antiderivative(lo)


def _polyfit_integral_inverse(metrics: np.ndarray, log_rates: np.ndarray, lo: float, hi: float) -> float:
    """
    Fit a 3rd-order polynomial log_rate = g(metric) and integrate over [lo, hi].

    Used for BD-Rate computation where we integrate log-rate as a function of quality.

    Args:
        metrics: Quality metric values (e.g. PSNR).
        log_rates: Log-transformed rate values.
        lo: Lower integration bound (in metric domain).
        hi: Upper integration bound (in metric domain).

    Returns:
        Definite integral of the fitted polynomial over [lo, hi].
    """
    # Fit: log_rate = a * psnr^3 + b * psnr^2 + c * psnr + d
    coeffs = np.polyfit(metrics, log_rates, deg=3)
    a, b, c, d = coeffs

    def antiderivative(x: float) -> float:
        return (a / 4.0) * x**4 + (b / 3.0) * x**3 + (c / 2.0) * x**2 + d * x

    return antiderivative(hi) - antiderivative(lo)


def calculate_bd_rate(
    rate_anchor: Union[List[float], np.ndarray],
    metric_anchor: Union[List[float], np.ndarray],
    rate_test: Union[List[float], np.ndarray],
    metric_test: Union[List[float], np.ndarray],
    min_points: int = 4,
) -> float:
    """
    Calculate Bjøntegaard Delta Rate (BD-Rate) in percent.

    Measures the average percentage difference in bitrate between a test codec
    and an anchor codec for equivalent quality (e.g., PSNR).

    Sign convention:
        Negative BD-Rate = test codec is MORE efficient (fewer bits for same quality).
        Positive BD-Rate = test codec is LESS efficient (more bits for same quality).

    Args:
        rate_anchor: Bitrates for the anchor codec (kbps, bpp, or any consistent unit).
        metric_anchor: Quality metric for the anchor codec (e.g. PSNR in dB).
        rate_test: Bitrates for the test codec.
        metric_test: Quality metric for the test codec.
        min_points: Minimum number of RD points required (default 4).

    Returns:
        BD-Rate in percent (e.g. -15.3 means 15.3% bitrate saving).

    Raises:
        ValueError: If inputs are invalid or integration range is degenerate.
    """
    r_a = np.asarray(rate_anchor, dtype=np.float64)
    m_a = np.asarray(metric_anchor, dtype=np.float64)
    r_t = np.asarray(rate_test, dtype=np.float64)
    m_t = np.asarray(metric_test, dtype=np.float64)

    r_a, m_a = _validate_rd_points(r_a, m_a, "Anchor", min_points)
    r_t, m_t = _validate_rd_points(r_t, m_t, "Test", min_points)

    # Use log of rate
    log_r_a = np.log(r_a)
    log_r_t = np.log(r_t)

    # Find common metric (PSNR) range
    psnr_lo = max(np.min(m_a), np.min(m_t))
    psnr_hi = min(np.max(m_a), np.max(m_t))

    if psnr_hi - psnr_lo < 1e-6:
        raise ValueError(
            f"No overlapping quality range between anchor [{np.min(m_a):.2f}, {np.max(m_a):.2f}] "
            f"and test [{np.min(m_t):.2f}, {np.max(m_t):.2f}]. "
            "BD-Rate requires a common quality interval for integration."
        )

    # Integrate log(rate) = f(psnr) for both anchor and test
    int_anchor = _polyfit_integral_inverse(m_a, log_r_a, psnr_lo, psnr_hi)
    int_test = _polyfit_integral_inverse(m_t, log_r_t, psnr_lo, psnr_hi)

    # BD-Rate = (10^((int_test - int_anchor) / (psnr_hi - psnr_lo)) - 1) * 100
    avg_diff = (int_test - int_anchor) / (psnr_hi - psnr_lo)
    bd_rate = (math.exp(avg_diff) - 1.0) * 100.0

    return bd_rate


def calculate_bd_psnr(
    rate_anchor: Union[List[float], np.ndarray],
    psnr_anchor: Union[List[float], np.ndarray],
    rate_test: Union[List[float], np.ndarray],
    psnr_test: Union[List[float], np.ndarray],
    min_points: int = 4,
) -> float:
    """
    Calculate Bjøntegaard Delta PSNR (BD-PSNR) in dB.

    Measures the average quality (PSNR) difference between a test codec
    and an anchor codec at equivalent bitrate.

    Sign convention:
        Positive BD-PSNR = test codec has HIGHER quality (improvement).
        Negative BD-PSNR = test codec has LOWER quality (degradation).

    Args:
        rate_anchor: Bitrates for the anchor codec.
        psnr_anchor: PSNR values for the anchor codec (dB).
        rate_test: Bitrates for the test codec.
        psnr_test: PSNR values for the test codec (dB).
        min_points: Minimum number of RD points required (default 4).

    Returns:
        BD-PSNR in dB.

    Raises:
        ValueError: If inputs are invalid or integration range is degenerate.
    """
    r_a = np.asarray(rate_anchor, dtype=np.float64)
    p_a = np.asarray(psnr_anchor, dtype=np.float64)
    r_t = np.asarray(rate_test, dtype=np.float64)
    p_t = np.asarray(psnr_test, dtype=np.float64)

    r_a, p_a = _validate_rd_points(r_a, p_a, "Anchor", min_points)
    r_t, p_t = _validate_rd_points(r_t, p_t, "Test", min_points)

    log_r_a = np.log(r_a)
    log_r_t = np.log(r_t)

    # Find common log-rate range
    log_rate_lo = max(np.min(log_r_a), np.min(log_r_t))
    log_rate_hi = min(np.max(log_r_a), np.max(log_r_t))

    if log_rate_hi - log_rate_lo < 1e-6:
        raise ValueError(
            f"No overlapping rate range between anchor [{np.min(r_a):.4f}, {np.max(r_a):.4f}] "
            f"and test [{np.min(r_t):.4f}, {np.max(r_t):.4f}]. "
            "BD-PSNR requires a common rate interval for integration."
        )

    # Integrate PSNR = f(log_rate) for both anchor and test
    int_anchor = _polyfit_integral(log_r_a, p_a, log_rate_lo, log_rate_hi)
    int_test = _polyfit_integral(log_r_t, p_t, log_rate_lo, log_rate_hi)

    # BD-PSNR = (int_test - int_anchor) / (log_rate_hi - log_rate_lo)
    bd_psnr = (int_test - int_anchor) / (log_rate_hi - log_rate_lo)

    return bd_psnr


def calculate_bd_metrics(
    rate_anchor: Union[List[float], np.ndarray],
    psnr_anchor: Union[List[float], np.ndarray],
    rate_test: Union[List[float], np.ndarray],
    psnr_test: Union[List[float], np.ndarray],
    anchor_name: str = "Anchor",
    test_name: str = "Test",
    min_points: int = 4,
) -> BDResult:
    """
    Calculate both BD-Rate and BD-PSNR in a single call.

    Args:
        rate_anchor: Bitrates for the anchor codec.
        psnr_anchor: PSNR values for the anchor codec.
        rate_test: Bitrates for the test codec.
        psnr_test: PSNR values for the test codec.
        anchor_name: Human-readable name for the anchor.
        test_name: Human-readable name for the test codec.
        min_points: Minimum points required.

    Returns:
        BDResult with bd_rate_percent and bd_psnr_db.
    """
    bd_rate = calculate_bd_rate(rate_anchor, psnr_anchor, rate_test, psnr_test, min_points)
    bd_psnr = calculate_bd_psnr(rate_anchor, psnr_anchor, rate_test, psnr_test, min_points)

    r_a = np.asarray(rate_anchor, dtype=np.float64)
    r_t = np.asarray(rate_test, dtype=np.float64)
    p_a = np.asarray(psnr_anchor, dtype=np.float64)
    p_t = np.asarray(psnr_test, dtype=np.float64)

    return BDResult(
        bd_rate_percent=bd_rate,
        bd_psnr_db=bd_psnr,
        anchor_codec=anchor_name,
        test_codec=test_name,
        num_anchor_points=len(r_a),
        num_test_points=len(r_t),
        common_rate_range=(
            max(float(np.min(r_a)), float(np.min(r_t))),
            min(float(np.max(r_a)), float(np.max(r_t))),
        ),
        common_psnr_range=(
            max(float(np.min(p_a)), float(np.min(p_t))),
            min(float(np.max(p_a)), float(np.max(p_t))),
        ),
    )

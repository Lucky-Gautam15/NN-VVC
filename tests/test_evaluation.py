"""
Phase F-1 Test Suite: Evaluation Metrics & Bjøntegaard Delta Infrastructure.

Tests MSE, PSNR, YUV-PSNR, MS-SSIM, Bitrate, BPP, BD-Rate, and BD-PSNR.
All tests use deterministic synthetic data with analytically predictable results.
"""

import math
import unittest
from typing import List, Tuple

import numpy as np
import torch

from src.evaluation.metrics import (
    RDPoint,
    SequenceMetrics,
    calculate_bitrate_bits_per_second,
    calculate_bpp,
    calculate_mse,
    calculate_ms_ssim,
    calculate_psnr,
    calculate_psnr_yuv,
)
from src.evaluation.bd_rate import (
    BDResult,
    calculate_bd_metrics,
    calculate_bd_psnr,
    calculate_bd_rate,
)


class TestMSE(unittest.TestCase):
    """Test Mean Squared Error calculation."""

    def test_identical_arrays_mse_zero(self):
        """Identical arrays should produce MSE = 0."""
        a = np.random.rand(64, 64).astype(np.float32)
        self.assertAlmostEqual(calculate_mse(a, a), 0.0, places=10)

    def test_known_mse_uint8(self):
        """Known constant difference on uint8 data should produce expected MSE."""
        ref = np.full((10, 10), 100, dtype=np.uint8)
        rec = np.full((10, 10), 110, dtype=np.uint8)
        # (110 - 100)^2 = 100.0
        self.assertAlmostEqual(calculate_mse(ref, rec), 100.0, places=6)

    def test_known_mse_float(self):
        """Float [0,1] data should produce correct MSE."""
        ref = np.full((4, 4), 0.5, dtype=np.float32)
        rec = np.full((4, 4), 0.7, dtype=np.float32)
        expected = (0.7 - 0.5) ** 2  # 0.04
        self.assertAlmostEqual(calculate_mse(ref, rec), expected, places=6)

    def test_pytorch_tensor_input(self):
        """PyTorch tensor input should work identically."""
        ref = torch.full((4, 4), 0.5)
        rec = torch.full((4, 4), 0.7)
        expected = (0.2) ** 2
        self.assertAlmostEqual(calculate_mse(ref, rec), expected, places=5)

    def test_shape_mismatch_raises(self):
        """Shape mismatch should raise ValueError."""
        ref = np.zeros((4, 4))
        rec = np.zeros((4, 5))
        with self.assertRaises(ValueError):
            calculate_mse(ref, rec)

    def test_empty_array_raises(self):
        """Empty arrays should raise ValueError."""
        ref = np.array([])
        rec = np.array([])
        with self.assertRaises(ValueError):
            calculate_mse(ref, rec)


class TestPSNR(unittest.TestCase):
    """Test Peak Signal-to-Noise Ratio calculation."""

    def test_identical_images_inf_psnr(self):
        """Identical images should produce inf PSNR."""
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        psnr = calculate_psnr(img, img, data_range=255.0)
        self.assertTrue(math.isinf(psnr))

    def test_known_psnr_8bit(self):
        """Known MSE on 8-bit data should produce expected PSNR."""
        ref = np.full((10, 10), 128, dtype=np.uint8)
        rec = np.full((10, 10), 138, dtype=np.uint8)
        # MSE = 100, PSNR = 10*log10(255^2/100) = 10*log10(650.25) ~= 28.13 dB
        expected = 10.0 * math.log10(255.0 ** 2 / 100.0)
        psnr = calculate_psnr(ref, rec, data_range=255.0)
        self.assertAlmostEqual(psnr, expected, places=2)

    def test_known_psnr_10bit(self):
        """10-bit data with known MSE should produce correct PSNR."""
        ref = np.full((8, 8), 512, dtype=np.uint16)
        rec = np.full((8, 8), 522, dtype=np.uint16)
        # MSE = 100, data_range = 1023
        expected = 10.0 * math.log10(1023.0 ** 2 / 100.0)
        psnr = calculate_psnr(ref, rec, data_range=1023.0)
        self.assertAlmostEqual(psnr, expected, places=2)

    def test_known_psnr_float01(self):
        """Float [0,1] data should produce correct PSNR."""
        ref = np.full((8, 8), 0.5, dtype=np.float32)
        rec = np.full((8, 8), 0.6, dtype=np.float32)
        # MSE = 0.01, data_range = 1.0
        expected = 10.0 * math.log10(1.0 / 0.01)  # 20.0 dB
        psnr = calculate_psnr(ref, rec, data_range=1.0)
        self.assertAlmostEqual(psnr, expected, places=2)

    def test_shape_mismatch_raises(self):
        """Shape mismatch should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_psnr(np.zeros((4, 4)), np.zeros((4, 5)), data_range=1.0)

    def test_negative_data_range_raises(self):
        """Negative data_range should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_psnr(np.zeros((4, 4)), np.zeros((4, 4)), data_range=-1.0)


class TestPSNRYUV(unittest.TestCase):
    """Test YUV420 PSNR calculation with proper component weighting."""

    def test_identical_yuv_inf_psnr(self):
        """Identical YUV420 frames should produce inf PSNR for all components."""
        y = np.full((64, 64), 128, dtype=np.uint8)
        u = np.full((32, 32), 128, dtype=np.uint8)
        v = np.full((32, 32), 128, dtype=np.uint8)
        result = calculate_psnr_yuv((y, u, v), (y, u, v), bit_depth=8)
        self.assertTrue(math.isinf(result["psnr_y"]))
        self.assertTrue(math.isinf(result["psnr_u"]))
        self.assertTrue(math.isinf(result["psnr_v"]))
        self.assertTrue(math.isinf(result["psnr_yuv"]))

    def test_known_yuv_mse(self):
        """Known constant offsets in Y/U/V should produce expected component PSNRs."""
        h, w = 64, 64
        y_ref = np.full((h, w), 128, dtype=np.uint8)
        u_ref = np.full((h // 2, w // 2), 128, dtype=np.uint8)
        v_ref = np.full((h // 2, w // 2), 128, dtype=np.uint8)

        y_rec = np.full((h, w), 138, dtype=np.uint8)  # MSE=100
        u_rec = np.full((h // 2, w // 2), 133, dtype=np.uint8)  # MSE=25
        v_rec = np.full((h // 2, w // 2), 130, dtype=np.uint8)  # MSE=4

        result = calculate_psnr_yuv(
            (y_ref, u_ref, v_ref), (y_rec, u_rec, v_rec), bit_depth=8
        )

        expected_y = 10.0 * math.log10(255.0 ** 2 / 100.0)
        expected_u = 10.0 * math.log10(255.0 ** 2 / 25.0)
        expected_v = 10.0 * math.log10(255.0 ** 2 / 4.0)
        expected_yuv = (6.0 * expected_y + expected_u + expected_v) / 8.0

        self.assertAlmostEqual(result["psnr_y"], expected_y, places=2)
        self.assertAlmostEqual(result["psnr_u"], expected_u, places=2)
        self.assertAlmostEqual(result["psnr_v"], expected_v, places=2)
        self.assertAlmostEqual(result["psnr_yuv"], expected_yuv, places=2)

    def test_10bit_yuv(self):
        """10-bit YUV data should compute correctly with data_range=1023."""
        y_ref = np.full((16, 16), 512, dtype=np.uint16)
        u_ref = np.full((8, 8), 512, dtype=np.uint16)
        v_ref = np.full((8, 8), 512, dtype=np.uint16)

        y_rec = np.full((16, 16), 522, dtype=np.uint16)
        u_rec = np.full((8, 8), 517, dtype=np.uint16)
        v_rec = np.full((8, 8), 514, dtype=np.uint16)

        result = calculate_psnr_yuv(
            (y_ref, u_ref, v_ref), (y_rec, u_rec, v_rec), bit_depth=10
        )
        expected_y = 10.0 * math.log10(1023.0 ** 2 / 100.0)
        self.assertAlmostEqual(result["psnr_y"], expected_y, places=2)
        self.assertGreater(result["psnr_y"], 0.0)

    def test_invalid_tuple_length(self):
        """Non-3-tuple should raise ValueError."""
        y = np.full((8, 8), 128, dtype=np.uint8)
        with self.assertRaises(ValueError):
            calculate_psnr_yuv((y, y), (y, y, y), bit_depth=8)

    def test_mse_values_returned(self):
        """MSE values should be present in returned dict."""
        y = np.full((8, 8), 100, dtype=np.uint8)
        u = np.full((4, 4), 100, dtype=np.uint8)
        v = np.full((4, 4), 100, dtype=np.uint8)
        result = calculate_psnr_yuv((y, u, v), (y, u, v), bit_depth=8)
        self.assertAlmostEqual(result["mse_y"], 0.0, places=10)
        self.assertAlmostEqual(result["mse_u"], 0.0, places=10)
        self.assertAlmostEqual(result["mse_v"], 0.0, places=10)


class TestMSSSIM(unittest.TestCase):
    """Test Multi-Scale Structural Similarity (MS-SSIM) calculation."""

    def test_identical_image_ms_ssim_approx_1(self):
        """Identical images should produce MS-SSIM approximately 1.0."""
        img = np.random.rand(64, 64).astype(np.float32)
        score = calculate_ms_ssim(img, img, data_range=1.0)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_degraded_image_lower_ms_ssim(self):
        """Adding noise should reduce MS-SSIM below 1.0."""
        np.random.seed(42)
        img = np.random.rand(64, 64).astype(np.float32)
        noisy = np.clip(img + np.random.randn(64, 64).astype(np.float32) * 0.3, 0, 1)
        score = calculate_ms_ssim(img, noisy, data_range=1.0)
        self.assertLess(score, 0.95)
        self.assertGreater(score, 0.0)

    def test_numpy_3d_input(self):
        """NumPy (H, W, C) input should be accepted."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        score = calculate_ms_ssim(img, img, data_range=1.0)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_pytorch_tensor_input(self):
        """PyTorch tensor (C, H, W) input should work."""
        t = torch.rand(3, 64, 64)
        score = calculate_ms_ssim(t, t, data_range=1.0)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_pytorch_4d_tensor_input(self):
        """PyTorch 4D tensor (1, C, H, W) should work."""
        t = torch.rand(1, 1, 64, 64)
        score = calculate_ms_ssim(t, t, data_range=1.0)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_shape_mismatch_raises(self):
        """Shape mismatch should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_ms_ssim(np.zeros((64, 64)), np.zeros((64, 32)), data_range=1.0)

    def test_small_image_fallback(self):
        """Small images that don't support 5 scales should still produce a valid score."""
        img = np.random.rand(16, 16).astype(np.float32)
        score = calculate_ms_ssim(img, img, data_range=1.0)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)


class TestBitrate(unittest.TestCase):
    """Test bitrate and bits-per-pixel calculations."""

    def test_known_bitrate_bps(self):
        """Known byte count at known fps should produce exact bps."""
        # 1000 bytes, 30 frames, 30 fps => duration 1s => 8000 bps
        bps = calculate_bitrate_bits_per_second(1000, 30, 30.0)
        self.assertAlmostEqual(bps, 8000.0, places=2)

    def test_known_bitrate_fractional_fps(self):
        """Fractional FPS should produce correct bitrate."""
        # 500 bytes, 15 frames, 29.97 fps => duration = 15/29.97 ~= 0.5005s => ~7993 bps
        bps = calculate_bitrate_bits_per_second(500, 15, 29.97)
        expected = (500 * 8) / (15 / 29.97)
        self.assertAlmostEqual(bps, expected, places=1)

    def test_known_bpp(self):
        """Known byte count with known dimensions should produce exact bpp."""
        # 512 bytes, 64x64 frame, 1 frame => bpp = 512*8/(64*64*1) = 1.0
        bpp = calculate_bpp(512, 64, 64, 1)
        self.assertAlmostEqual(bpp, 1.0, places=6)

    def test_bpp_multi_frame(self):
        """Multi-frame bpp should divide by total pixels across frames."""
        # 2048 bytes, 32x32, 4 frames => bpp = 2048*8 / (32*32*4) = 4.0
        bpp = calculate_bpp(2048, 32, 32, 4)
        self.assertAlmostEqual(bpp, 4.0, places=6)

    def test_zero_bytes_produces_zero(self):
        """Zero-byte stream should produce zero bps and zero bpp."""
        self.assertAlmostEqual(calculate_bitrate_bits_per_second(0, 10, 30.0), 0.0)
        self.assertAlmostEqual(calculate_bpp(0, 64, 64, 1), 0.0)

    def test_invalid_fps_raises(self):
        """Non-positive FPS should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_bitrate_bits_per_second(100, 10, 0.0)
        with self.assertRaises(ValueError):
            calculate_bitrate_bits_per_second(100, 10, -1.0)

    def test_invalid_frame_count_raises(self):
        """Non-positive frame count should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_bitrate_bits_per_second(100, 0, 30.0)
        with self.assertRaises(ValueError):
            calculate_bpp(100, 64, 64, 0)

    def test_invalid_dimensions_raises(self):
        """Non-positive dimensions should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_bpp(100, 0, 64, 1)
        with self.assertRaises(ValueError):
            calculate_bpp(100, 64, -1, 1)

    def test_negative_bytes_raises(self):
        """Negative byte count should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_bitrate_bits_per_second(-1, 10, 30.0)
        with self.assertRaises(ValueError):
            calculate_bpp(-1, 64, 64, 1)


class TestBDRate(unittest.TestCase):
    """Test Bjøntegaard Delta Rate calculation."""

    def _make_identical_curves(self):
        """Create identical anchor and test RD curves for baseline tests."""
        rates = [100, 200, 400, 800]
        psnrs = [30.0, 33.0, 36.0, 39.0]
        return rates, psnrs, rates[:], psnrs[:]

    def test_identical_curves_bd_rate_zero(self):
        """Identical RD curves should produce approximately 0% BD-Rate."""
        r_a, p_a, r_t, p_t = self._make_identical_curves()
        bd_rate = calculate_bd_rate(r_a, p_a, r_t, p_t)
        self.assertAlmostEqual(bd_rate, 0.0, places=2)

    def test_better_codec_negative_bd_rate(self):
        """Test codec with same PSNR at lower rates should produce negative BD-Rate."""
        r_a = [100, 200, 400, 800]
        p_a = [30.0, 33.0, 36.0, 39.0]
        # Test codec: same quality at half the bitrate
        r_t = [50, 100, 200, 400]
        p_t = [30.0, 33.0, 36.0, 39.0]
        bd_rate = calculate_bd_rate(r_a, p_a, r_t, p_t)
        self.assertLess(bd_rate, 0.0)
        # Should be approximately -50%
        self.assertAlmostEqual(bd_rate, -50.0, delta=5.0)

    def test_worse_codec_positive_bd_rate(self):
        """Test codec needing more bits for same quality should produce positive BD-Rate."""
        r_a = [100, 200, 400, 800]
        p_a = [30.0, 33.0, 36.0, 39.0]
        # Test codec: same quality at double the bitrate
        r_t = [200, 400, 800, 1600]
        p_t = [30.0, 33.0, 36.0, 39.0]
        bd_rate = calculate_bd_rate(r_a, p_a, r_t, p_t)
        self.assertGreater(bd_rate, 0.0)

    def test_qp_order_independence(self):
        """BD-Rate result should be independent of QP ordering in input."""
        r_a = [800, 100, 400, 200]
        p_a = [39.0, 30.0, 36.0, 33.0]
        r_t = [400, 50, 200, 100]
        p_t = [39.0, 30.0, 36.0, 33.0]

        r_a_sorted = [100, 200, 400, 800]
        p_a_sorted = [30.0, 33.0, 36.0, 39.0]
        r_t_sorted = [50, 100, 200, 400]
        p_t_sorted = [30.0, 33.0, 36.0, 39.0]

        bd_unsorted = calculate_bd_rate(r_a, p_a, r_t, p_t)
        bd_sorted = calculate_bd_rate(r_a_sorted, p_a_sorted, r_t_sorted, p_t_sorted)
        self.assertAlmostEqual(bd_unsorted, bd_sorted, places=4)

    def test_insufficient_points_raises(self):
        """Fewer than 4 RD points should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_bd_rate([100, 200, 400], [30, 33, 36], [100, 200, 400], [30, 33, 36])

    def test_duplicate_rate_rejection(self):
        """Duplicate rate values should be averaged, but if duplicates reduce unique count below 4, raise."""
        # 4 points with all same rate = only 1 unique point
        with self.assertRaises(ValueError):
            calculate_bd_rate(
                [100, 100, 100, 100], [30, 33, 36, 39],
                [100, 200, 400, 800], [30, 33, 36, 39],
            )

    def test_non_overlapping_psnr_raises(self):
        """Non-overlapping quality ranges should raise ValueError."""
        r_a = [100, 200, 400, 800]
        p_a = [30.0, 33.0, 36.0, 39.0]
        r_t = [100, 200, 400, 800]
        p_t = [50.0, 53.0, 56.0, 59.0]  # Completely above anchor range
        with self.assertRaises(ValueError):
            calculate_bd_rate(r_a, p_a, r_t, p_t)

    def test_negative_rate_raises(self):
        """Negative rate values should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_bd_rate([-100, 200, 400, 800], [30, 33, 36, 39],
                              [100, 200, 400, 800], [30, 33, 36, 39])

    def test_nan_values_rejected(self):
        """NaN values in rates or metrics should raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_bd_rate(
                [100, float("nan"), 400, 800], [30, 33, 36, 39],
                [100, 200, 400, 800], [30, 33, 36, 39],
            )


class TestBDPSNR(unittest.TestCase):
    """Test Bjøntegaard Delta PSNR calculation."""

    def test_identical_curves_bd_psnr_zero(self):
        """Identical curves should produce approximately 0 dB BD-PSNR."""
        rates = [100, 200, 400, 800]
        psnrs = [30.0, 33.0, 36.0, 39.0]
        bd_psnr = calculate_bd_psnr(rates, psnrs, rates[:], psnrs[:])
        self.assertAlmostEqual(bd_psnr, 0.0, places=2)

    def test_better_codec_positive_bd_psnr(self):
        """Test codec with higher quality at same rate should produce positive BD-PSNR."""
        r_a = [100, 200, 400, 800]
        p_a = [30.0, 33.0, 36.0, 39.0]
        r_t = [100, 200, 400, 800]
        p_t = [32.0, 35.0, 38.0, 41.0]  # +2 dB at each rate
        bd_psnr = calculate_bd_psnr(r_a, p_a, r_t, p_t)
        self.assertGreater(bd_psnr, 0.0)
        self.assertAlmostEqual(bd_psnr, 2.0, delta=0.5)

    def test_worse_codec_negative_bd_psnr(self):
        """Test codec with lower quality at same rate should produce negative BD-PSNR."""
        r_a = [100, 200, 400, 800]
        p_a = [30.0, 33.0, 36.0, 39.0]
        r_t = [100, 200, 400, 800]
        p_t = [28.0, 31.0, 34.0, 37.0]  # -2 dB at each rate
        bd_psnr = calculate_bd_psnr(r_a, p_a, r_t, p_t)
        self.assertLess(bd_psnr, 0.0)

    def test_qp_order_independence(self):
        """BD-PSNR should not depend on input ordering."""
        r_a = [800, 100, 400, 200]
        p_a = [39.0, 30.0, 36.0, 33.0]
        r_t = [800, 100, 400, 200]
        p_t = [41.0, 32.0, 38.0, 35.0]

        r_a_s = [100, 200, 400, 800]
        p_a_s = [30.0, 33.0, 36.0, 39.0]
        r_t_s = [100, 200, 400, 800]
        p_t_s = [32.0, 35.0, 38.0, 41.0]

        bd_unsorted = calculate_bd_psnr(r_a, p_a, r_t, p_t)
        bd_sorted = calculate_bd_psnr(r_a_s, p_a_s, r_t_s, p_t_s)
        self.assertAlmostEqual(bd_unsorted, bd_sorted, places=4)

    def test_non_overlapping_rate_raises(self):
        """Non-overlapping rate ranges should raise ValueError."""
        r_a = [100, 200, 400, 800]
        p_a = [30.0, 33.0, 36.0, 39.0]
        r_t = [2000, 3000, 4000, 5000]  # Completely above anchor range
        p_t = [30.0, 33.0, 36.0, 39.0]
        with self.assertRaises(ValueError):
            calculate_bd_psnr(r_a, p_a, r_t, p_t)


class TestBDMetrics(unittest.TestCase):
    """Test combined BD-Rate + BD-PSNR calculation."""

    def test_bd_metrics_returns_both(self):
        """calculate_bd_metrics should return BDResult with both bd_rate and bd_psnr."""
        r_a = [100, 200, 400, 800]
        p_a = [30.0, 33.0, 36.0, 39.0]
        r_t = [50, 100, 200, 400]
        p_t = [30.0, 33.0, 36.0, 39.0]
        result = calculate_bd_metrics(r_a, p_a, r_t, p_t, "VTM", "NN-VVC")
        self.assertIsInstance(result, BDResult)
        self.assertLess(result.bd_rate_percent, 0.0)
        self.assertGreater(result.bd_psnr_db, 0.0)
        self.assertEqual(result.anchor_codec, "VTM")
        self.assertEqual(result.test_codec, "NN-VVC")
        self.assertEqual(result.num_anchor_points, 4)
        self.assertEqual(result.num_test_points, 4)
        self.assertIsNotNone(result.common_rate_range)
        self.assertIsNotNone(result.common_psnr_range)


class TestRDPointDataclass(unittest.TestCase):
    """Test RDPoint and SequenceMetrics dataclasses."""

    def test_rd_point_creation(self):
        """RDPoint should be constructible with required fields."""
        pt = RDPoint(qp=32, bitrate_kbps=1500.0, bpp=0.5, total_bits=120000)
        self.assertEqual(pt.qp, 32)
        self.assertAlmostEqual(pt.bitrate_kbps, 1500.0)
        self.assertIsNone(pt.psnr_y)

    def test_sequence_metrics_creation(self):
        """SequenceMetrics should hold a list of RD points."""
        pts = [
            RDPoint(qp=22, bitrate_kbps=5000.0, bpp=1.5, total_bits=400000, psnr_y=42.0),
            RDPoint(qp=37, bitrate_kbps=500.0, bpp=0.15, total_bits=40000, psnr_y=35.0),
        ]
        seq = SequenceMetrics(
            sequence_name="BasketballDrill",
            width=832,
            height=480,
            frame_count=500,
            framerate=50.0,
            rd_points=pts,
        )
        self.assertEqual(seq.sequence_name, "BasketballDrill")
        self.assertEqual(len(seq.rd_points), 2)


if __name__ == "__main__":
    unittest.main()

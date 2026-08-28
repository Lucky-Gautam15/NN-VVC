import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.vvc.yuv_utils import (
    calculate_yuv420_frame_bytes,
    get_yuv_frame_count,
    pad_to_alignment,
    read_yuv_frame,
    read_yuv_sequence,
    rgb_to_yuv420,
    tensor_to_yuv420_bytes,
    unpad_from_alignment,
    write_yuv_frame,
    write_yuv_sequence,
    yuv420_bytes_to_tensor,
    yuv420_to_rgb,
)
from src.vvc.vtm_wrapper import VTMWrapper


class TestYUVUtils(unittest.TestCase):
    """
    Test suite for Phase E-2 YUV and spatial utilities.
    Validates RGB <-> YUV420 conversions, BT.601/BT.709 standards,
    planar file I/O, padding, tensor bridges, and VTM 12.0 planar compatibility.
    """

    @classmethod
    def setUpClass(cls):
        # Configure temporary test directory on E:\temp if available
        temp_base = Path(r"E:\temp") if Path(r"E:\temp").is_dir() else None
        cls.temp_dir = tempfile.mkdtemp(prefix="nnvvc_yuv_test_", dir=temp_base)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio between two RGB images in [0, 1]."""
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10(1.0 / mse))

    # -------------------------------------------------------------------------
    # 1. RGB -> YUV Shape Correctness
    # -------------------------------------------------------------------------
    def test_01_rgb_to_yuv_shapes(self):
        """Verify planar Y, U, V array dimensions for YUV420."""
        h, w = 64, 128
        rgb = np.random.rand(h, w, 3).astype(np.float32)
        y, u, v = rgb_to_yuv420(rgb, standard="bt709", bit_depth=8)

        self.assertEqual(y.shape, (h, w))
        self.assertEqual(u.shape, (h // 2, w // 2))
        self.assertEqual(v.shape, (h // 2, w // 2))
        self.assertEqual(y.dtype, np.uint8)
        self.assertEqual(u.dtype, np.uint8)
        self.assertEqual(v.dtype, np.uint8)

    # -------------------------------------------------------------------------
    # 2. YUV -> RGB Shape Correctness
    # -------------------------------------------------------------------------
    def test_02_yuv_to_rgb_shapes(self):
        """Verify RGB output shape and dtype from planar YUV420."""
        h, w = 64, 64
        y = np.full((h, w), 128, dtype=np.uint8)
        u = np.full((h // 2, w // 2), 128, dtype=np.uint8)
        v = np.full((h // 2, w // 2), 128, dtype=np.uint8)

        rgb_np = yuv420_to_rgb(y, u, v, standard="bt709", return_tensor=False)
        self.assertEqual(rgb_np.shape, (h, w, 3))
        self.assertEqual(rgb_np.dtype, np.float32)
        self.assertTrue((rgb_np >= 0.0).all() and (rgb_np <= 1.0).all())

        rgb_tensor = yuv420_to_rgb(y, u, v, standard="bt709", return_tensor=True)
        self.assertEqual(rgb_tensor.shape, (3, h, w))
        self.assertEqual(rgb_tensor.dtype, torch.float32)

    # -------------------------------------------------------------------------
    # 3. BT.601 Conversion Accuracy
    # -------------------------------------------------------------------------
    def test_03_bt601_conversion(self):
        """Verify BT.601 standard matrix values for primary colors."""
        # Pure White [1, 1, 1] -> Y=255, U=128, V=128
        white = np.ones((16, 16, 3), dtype=np.float32)
        y, u, v = rgb_to_yuv420(white, standard="bt601", bit_depth=8)
        self.assertEqual(int(y[0, 0]), 255)
        self.assertEqual(int(u[0, 0]), 128)
        self.assertEqual(int(v[0, 0]), 128)

        # Pure Black [0, 0, 0] -> Y=0, U=128, V=128
        black = np.zeros((16, 16, 3), dtype=np.float32)
        y, u, v = rgb_to_yuv420(black, standard="bt601", bit_depth=8)
        self.assertEqual(int(y[0, 0]), 0)
        self.assertEqual(int(u[0, 0]), 128)
        self.assertEqual(int(v[0, 0]), 128)

    # -------------------------------------------------------------------------
    # 4. BT.709 Conversion Accuracy
    # -------------------------------------------------------------------------
    def test_04_bt709_conversion(self):
        """Verify BT.709 standard matrix values for primary colors."""
        # Pure Green [0, 1, 0]: BT.709 Y = 0.7152 * 255 = ~182
        green = np.zeros((16, 16, 3), dtype=np.float32)
        green[:, :, 1] = 1.0
        y, u, v = rgb_to_yuv420(green, standard="bt709", bit_depth=8)
        self.assertAlmostEqual(float(y[0, 0]), 182.0, delta=2.0)

    # -------------------------------------------------------------------------
    # 5. RGB <-> YUV Roundtrip PSNR
    # -------------------------------------------------------------------------
    def test_05_rgb_yuv_roundtrip_psnr(self):
        """Verify RGB -> YUV420 -> RGB roundtrip preserves high visual fidelity."""
        # Create a smooth color gradient
        h, w = 64, 64
        r = np.linspace(0.1, 0.9, w).reshape(1, w).repeat(h, axis=0)
        g = np.linspace(0.2, 0.8, h).reshape(h, 1).repeat(w, axis=1)
        b = np.full((h, w), 0.5)
        rgb_orig = np.stack([r, g, b], axis=-1).astype(np.float32)

        # Roundtrip with BT.709
        y, u, v = rgb_to_yuv420(rgb_orig, standard="bt709")
        rgb_recon = yuv420_to_rgb(y, u, v, standard="bt709")

        psnr = self.compute_psnr(rgb_orig, rgb_recon)
        self.assertGreater(psnr, 35.0, f"Roundtrip PSNR {psnr:.2f} dB is lower than expected 35 dB threshold")

    # -------------------------------------------------------------------------
    # 6. Single-Frame Planar I/O Integrity
    # -------------------------------------------------------------------------
    def test_06_single_frame_io_integrity(self):
        """Verify writing and reading a single YUV420 frame produces identical bytes."""
        h, w = 32, 32
        y_in = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        u_in = np.random.randint(0, 256, (h // 2, w // 2), dtype=np.uint8)
        v_in = np.random.randint(0, 256, (h // 2, w // 2), dtype=np.uint8)

        file_path = Path(self.temp_dir) / "test_frame_io.yuv"
        write_yuv_frame(file_path, y_in, u_in, v_in, bit_depth=8)

        y_out, u_out, v_out = read_yuv_frame(file_path, w, h, frame_idx=0, bit_depth=8)

        np.testing.assert_array_equal(y_in, y_out)
        np.testing.assert_array_equal(u_in, u_out)
        np.testing.assert_array_equal(v_in, v_out)

    # -------------------------------------------------------------------------
    # 7. Multi-Frame Sequence I/O
    # -------------------------------------------------------------------------
    def test_07_sequence_io(self):
        """Verify writing and reading a multi-frame sequence preserves each frame."""
        h, w, n_frames = 32, 32, 4
        frames_in = []
        for i in range(n_frames):
            y = np.full((h, w), i * 50, dtype=np.uint8)
            u = np.full((h // 2, w // 2), 128 + i * 10, dtype=np.uint8)
            v = np.full((h // 2, w // 2), 128 - i * 10, dtype=np.uint8)
            frames_in.append((y, u, v))

        file_path = Path(self.temp_dir) / "test_seq_io.yuv"
        write_yuv_sequence(file_path, frames_in, w, h, bit_depth=8)

        frames_out = read_yuv_sequence(file_path, w, h, bit_depth=8)
        self.assertEqual(len(frames_out), n_frames)

        for i in range(n_frames):
            np.testing.assert_array_equal(frames_in[i][0], frames_out[i][0])
            np.testing.assert_array_equal(frames_in[i][1], frames_out[i][1])
            np.testing.assert_array_equal(frames_in[i][2], frames_out[i][2])

    # -------------------------------------------------------------------------
    # 8. Frame Count Calculation
    # -------------------------------------------------------------------------
    def test_08_frame_count_calculation(self):
        """Verify get_yuv_frame_count correctly counts frames in file."""
        h, w, n_frames = 64, 64, 5
        file_path = Path(self.temp_dir) / "test_count.yuv"
        frame_bytes = calculate_yuv420_frame_bytes(w, h, bit_depth=8)
        with open(file_path, "wb") as f:
            f.write(b"\x80" * (frame_bytes * n_frames))

        detected_count = get_yuv_frame_count(file_path, w, h, bit_depth=8)
        self.assertEqual(detected_count, n_frames)

    # -------------------------------------------------------------------------
    # 9. Frame Byte Sizing
    # -------------------------------------------------------------------------
    def test_09_frame_byte_sizing(self):
        """Verify exact byte calculation for 8-bit and 10-bit YUV420 frames."""
        w, h = 1920, 1080
        # 8-bit: 1920*1080*1.5 = 3,110,400 bytes
        self.assertEqual(calculate_yuv420_frame_bytes(w, h, bit_depth=8), 3110400)
        # 10-bit: 1920*1080*1.5 * 2 = 6,220,800 bytes
        self.assertEqual(calculate_yuv420_frame_bytes(w, h, bit_depth=10), 6220800)

        # Invalid dimensions must raise ValueError
        with self.assertRaises(ValueError):
            calculate_yuv420_frame_bytes(1921, 1080)
        with self.assertRaises(ValueError):
            calculate_yuv420_frame_bytes(1920, -1080)

    # -------------------------------------------------------------------------
    # 10. Spatial Padding and Unpadding (Even and 16/32 alignment)
    # -------------------------------------------------------------------------
    def test_10_padding_and_unpadding(self):
        """Verify spatial padding to alignment and reversible unpadding for NumPy and PyTorch."""
        # Odd dimensions NumPy
        orig_np = np.random.rand(53, 71, 3).astype(np.float32)
        padded_np, orig_shape = pad_to_alignment(orig_np, align=16, mode="reflect")
        self.assertEqual(padded_np.shape, (64, 80, 3))
        self.assertEqual(orig_shape, (53, 71))

        unpadded_np = unpad_from_alignment(padded_np, orig_shape)
        np.testing.assert_array_equal(orig_np, unpadded_np)

        # Odd dimensions PyTorch Tensor
        orig_t = torch.rand(3, 47, 59)
        padded_t, orig_shape_t = pad_to_alignment(orig_t, align=32, mode="reflect")
        self.assertEqual(padded_t.shape, (3, 64, 64))
        self.assertEqual(orig_shape_t, (47, 59))

        unpadded_t = unpad_from_alignment(padded_t, orig_shape_t)
        self.assertTrue(torch.equal(orig_t, unpadded_t))

    # -------------------------------------------------------------------------
    # 11. PyTorch Tensor <-> Raw YUV Bytes Bridges
    # -------------------------------------------------------------------------
    def test_11_tensor_bridges(self):
        """Verify tensor_to_yuv420_bytes and yuv420_bytes_to_tensor roundtrip."""
        t_in = torch.rand(3, 32, 32)
        raw_bytes = tensor_to_yuv420_bytes(t_in, standard="bt709", bit_depth=8)
        expected_bytes = calculate_yuv420_frame_bytes(32, 32, bit_depth=8)
        self.assertEqual(len(raw_bytes), expected_bytes)

        t_out = yuv420_bytes_to_tensor(raw_bytes, 32, 32, standard="bt709", bit_depth=8)
        self.assertEqual(t_out.shape, (3, 32, 32))
        self.assertEqual(t_out.dtype, torch.float32)

    # -------------------------------------------------------------------------
    # 12. Compatibility with VTM 12.0 Codec
    # -------------------------------------------------------------------------
    def test_12_vtm_codec_compatibility(self):
        """Verify that a sequence generated by yuv_utils encodes cleanly with real VTM 12.0."""
        vtm = VTMWrapper()
        w, h, n_frames = 64, 64, 2
        
        # Generate gradient sequence via rgb_to_yuv420
        frames = []
        for i in range(n_frames):
            rgb = np.full((h, w, 3), (i + 1) * 0.4, dtype=np.float32)
            frames.append(rgb_to_yuv420(rgb, standard="bt709", bit_depth=8))

        yuv_input_path = Path(self.temp_dir) / "vtm_compat_in.yuv"
        write_yuv_sequence(yuv_input_path, frames, w, h, bit_depth=8)

        bitstream_path = Path(self.temp_dir) / "vtm_compat.vvc"
        recon_path = Path(self.temp_dir) / "vtm_compat_recon.yuv"

        result = vtm.encode(
            input_yuv=yuv_input_path,
            width=w,
            height=h,
            frame_count=n_frames,
            qp=32,
            output_bitstream=bitstream_path,
            recon_yuv=recon_path,
        )

        self.assertEqual(result.returncode, 0)
        self.assertTrue(bitstream_path.is_file())
        self.assertGreater(bitstream_path.stat().st_size, 0)
        self.assertTrue(recon_path.is_file())
        self.assertEqual(recon_path.stat().st_size, calculate_yuv420_frame_bytes(w, h) * n_frames)


if __name__ == "__main__":
    unittest.main()

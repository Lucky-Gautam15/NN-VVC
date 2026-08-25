import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.vvc.vtm_wrapper import (
    VTMWrapper,
    VTMError,
    VTMNotAvailableError,
    VTMVersionError,
    VTMExecutionError,
    VTMEncodeResult,
    VTMDecodeResult,
)


class TestVTMWrapper(unittest.TestCase):
    """
    Test suite for Phase E-1 VTM 12.0 Python wrapper.
    Executes real VTM 12.0 binaries for encoding and decoding tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.vtm = VTMWrapper()
        cls.temp_dir = tempfile.mkdtemp(prefix="nnvvc_vtm_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def create_synthetic_yuv(
        self, width: int = 64, height: int = 64, frame_count: int = 2
    ) -> Path:
        """Create a deterministic synthetic 8-bit YUV420 file."""
        yuv_path = Path(self.temp_dir) / f"synthetic_{width}x{height}_{frame_count}f.yuv"
        with open(yuv_path, "wb") as f:
            for f_idx in range(frame_count):
                y = np.fromfunction(
                    lambda r, c: ((r * 4 + c * 4 + f_idx * 20) % 256).astype(np.uint8),
                    (height, width),
                )
                u = np.full((height // 2, width // 2), 128, dtype=np.uint8)
                v = np.full((height // 2, width // 2), 128, dtype=np.uint8)
                f.write(y.tobytes())
                f.write(u.tobytes())
                f.write(v.tobytes())
        return yuv_path

    # -------------------------------------------------------------------------
    # Test A: Executable Discovery
    # -------------------------------------------------------------------------
    def test_a_executable_discovery(self):
        """Verify that default VTM 12.0 executables exist and are recognized."""
        self.assertTrue(self.vtm.encoder_path.is_file(), f"EncoderApp not found at {self.vtm.encoder_path}")
        self.assertTrue(self.vtm.decoder_path.is_file(), f"DecoderApp not found at {self.vtm.decoder_path}")
        self.assertTrue(self.vtm.cfg_dir.is_dir(), f"Config directory not found at {self.vtm.cfg_dir}")

    # -------------------------------------------------------------------------
    # Test B: Version Validation
    # -------------------------------------------------------------------------
    def test_b_version_validation(self):
        """Verify that both encoder and decoder report version 12.0."""
        enc_ver = self.vtm.get_encoder_version()
        dec_ver = self.vtm.get_decoder_version()

        self.assertIn("12.0", enc_ver)
        self.assertIn("12.0", dec_ver)

    # -------------------------------------------------------------------------
    # Test C: Missing Executable & Error Handling
    # -------------------------------------------------------------------------
    def test_c_missing_executable_error(self):
        """Verify that initializing with non-existent binaries raises VTMNotAvailableError."""
        fake_path = Path(self.temp_dir) / "non_existent_encoder.exe"
        with self.assertRaises(VTMNotAvailableError):
            VTMWrapper(encoder_path=fake_path, auto_validate=True)

    # -------------------------------------------------------------------------
    # Test D: Real VTM Encoding
    # -------------------------------------------------------------------------
    def test_d_real_vtm_encode(self):
        """Run real VTM 12.0 encoding on synthetic YUV and verify output bitstream."""
        width, height, frames = 64, 64, 2
        input_yuv = self.create_synthetic_yuv(width, height, frames)
        bitstream_path = Path(self.temp_dir) / "test_d.vvc"
        recon_path = Path(self.temp_dir) / "test_d_recon_enc.yuv"

        result = self.vtm.encode(
            input_yuv=input_yuv,
            width=width,
            height=height,
            frame_count=frames,
            qp=32,
            output_bitstream=bitstream_path,
            recon_yuv=recon_path,
            cfg_name_or_path="encoder_lowdelay_P_vtm.cfg",
        )

        self.assertIsInstance(result, VTMEncodeResult)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(bitstream_path.is_file())
        self.assertGreater(bitstream_path.stat().st_size, 0)
        self.assertTrue(recon_path.is_file())
        expected_recon_bytes = width * height * 3 // 2 * frames
        self.assertEqual(recon_path.stat().st_size, expected_recon_bytes)
        self.assertIsNotNone(result.total_bits)
        self.assertGreater(result.total_bits, 0)

    # -------------------------------------------------------------------------
    # Test E: Real VTM Decoding
    # -------------------------------------------------------------------------
    def test_e_real_vtm_decode(self):
        """Run real VTM 12.0 decoding on a valid bitstream and verify reconstruction."""
        width, height, frames = 64, 64, 2
        input_yuv = self.create_synthetic_yuv(width, height, frames)
        bitstream_path = Path(self.temp_dir) / "test_e.vvc"
        recon_dec_path = Path(self.temp_dir) / "test_e_recon_dec.yuv"

        # First encode to get valid bitstream
        self.vtm.encode(
            input_yuv=input_yuv,
            width=width,
            height=height,
            frame_count=frames,
            qp=27,
            output_bitstream=bitstream_path,
        )

        # Decode bitstream
        dec_result = self.vtm.decode(
            bitstream_path=bitstream_path,
            output_recon_path=recon_dec_path,
            output_bit_depth=8,
        )

        self.assertIsInstance(dec_result, VTMDecodeResult)
        self.assertEqual(dec_result.returncode, 0)
        self.assertTrue(recon_dec_path.is_file())
        expected_recon_bytes = width * height * 3 // 2 * frames
        self.assertEqual(recon_dec_path.stat().st_size, expected_recon_bytes)

    # -------------------------------------------------------------------------
    # Test F: Reconstruction Consistency (Encoder Recon == Decoder Recon)
    # -------------------------------------------------------------------------
    def test_f_reconstruction_consistency(self):
        """Verify that encoder-generated reconstruction matches decoder-generated reconstruction bit-for-bit."""
        width, height, frames = 64, 64, 3
        input_yuv = self.create_synthetic_yuv(width, height, frames)
        bitstream_path = Path(self.temp_dir) / "test_f.vvc"
        recon_enc_path = Path(self.temp_dir) / "test_f_recon_enc.yuv"
        recon_dec_path = Path(self.temp_dir) / "test_f_recon_dec.yuv"

        self.vtm.encode(
            input_yuv=input_yuv,
            width=width,
            height=height,
            frame_count=frames,
            qp=22,
            output_bitstream=bitstream_path,
            recon_yuv=recon_enc_path,
        )

        self.vtm.decode(
            bitstream_path=bitstream_path,
            output_recon_path=recon_dec_path,
        )

        with open(recon_enc_path, "rb") as f_enc, open(recon_dec_path, "rb") as f_dec:
            enc_data = f_enc.read()
            dec_data = f_dec.read()
            self.assertEqual(
                enc_data,
                dec_data,
                "Encoder internal reconstruction does not match decoder reconstruction!",
            )

    # -------------------------------------------------------------------------
    # Test G: Invalid Parameters & Robustness
    # -------------------------------------------------------------------------
    def test_g_invalid_parameters_handling(self):
        """Verify invalid dimensions, frame counts, and QPs are cleanly rejected."""
        valid_yuv = self.create_synthetic_yuv(64, 64, 2)
        out_bit = Path(self.temp_dir) / "invalid_test.vvc"

        # Odd width
        with self.assertRaises(ValueError):
            self.vtm.encode(valid_yuv, width=63, height=64, frame_count=2, qp=32, output_bitstream=out_bit)

        # Negative height
        with self.assertRaises(ValueError):
            self.vtm.encode(valid_yuv, width=64, height=-64, frame_count=2, qp=32, output_bitstream=out_bit)

        # Zero frames
        with self.assertRaises(ValueError):
            self.vtm.encode(valid_yuv, width=64, height=64, frame_count=0, qp=32, output_bitstream=out_bit)

        # Out-of-bound QP
        with self.assertRaises(ValueError):
            self.vtm.encode(valid_yuv, width=64, height=64, frame_count=2, qp=64, output_bitstream=out_bit)

        # Non-existent input file
        with self.assertRaises(FileNotFoundError):
            self.vtm.encode(Path(self.temp_dir) / "no_such_file.yuv", width=64, height=64, frame_count=2, qp=32, output_bitstream=out_bit)

    # -------------------------------------------------------------------------
    # Test H: Output Metrics Parsing
    # -------------------------------------------------------------------------
    def test_h_stdout_parsing(self):
        """Verify stdout parsing extracts frame counts, bitrate, and PSNR accurately."""
        sample_stdout = """
POC    0 LId:  0 TId: 0 ( IDR_N_LP, I-SLICE, QP 26 )       2096 bits [Y 51.8489 dB    U 999.9900 dB    V 999.9900 dB] [ET     0 ] [L0] [L1]
POC    1 LId:  0 TId: 0 ( TRAIL, P-SLICE, QP 34 )        200 bits [Y 51.2117 dB    U 999.9900 dB    V 999.9900 dB] [ET     0 ] [L0 0c] [L1]

LayerId  0
	Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR   
	        2    a      11.4800   51.5303  999.9900  999.9900   53.2913

 finished @ Tue Aug 25 21:26:14 2026
 Total Time:        1.798 sec. [user]        1.799 sec. [elapsed]
"""
        metrics = VTMWrapper.parse_encoder_stdout(sample_stdout)
        self.assertEqual(metrics.get("frames_encoded"), 2)
        self.assertEqual(metrics.get("total_bits"), 2296)
        self.assertAlmostEqual(metrics.get("bitrate_kbps"), 11.48, places=2)
        self.assertAlmostEqual(metrics.get("psnr_y"), 51.5303, places=4)
        self.assertAlmostEqual(metrics.get("psnr_yuv"), 53.2913, places=4)
        self.assertAlmostEqual(metrics.get("encode_time_sec"), 1.799, places=3)


if __name__ == "__main__":
    unittest.main()

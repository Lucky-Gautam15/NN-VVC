import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.vvc.reference_injector import ReferenceInjector
from src.vvc.vtm_wrapper import VTMWrapper, VTMEncodeResult, VTMDecodeResult
from src.vvc.yuv_utils import (
    calculate_yuv420_frame_bytes,
    get_yuv_frame_count,
    read_yuv_frame,
    read_yuv_sequence,
    rgb_to_yuv420,
    write_yuv_sequence,
)


class TestReferenceInjector(unittest.TestCase):
    """
    Test suite for Phase E-3 ReferenceInjector.
    Validates neural I-frame reference sequence building, paper QP pairing,
    and real VTM 12.0 Low-Delay P reference injection encoding/decoding.
    """

    @classmethod
    def setUpClass(cls):
        temp_base = Path(r"E:\temp") if Path(r"E:\temp").is_dir() else None
        cls.temp_dir = tempfile.mkdtemp(prefix="nnvvc_ref_test_", dir=temp_base)
        cls.vtm = VTMWrapper()
        cls.injector = ReferenceInjector(vtm_wrapper=cls.vtm, temp_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. QP Pairing Verification (QP_intra = QP_inter - 5)
    # -------------------------------------------------------------------------
    def test_01_qp_pairing(self):
        """Verify paper-specified QP_intra = QP_inter - 5 calculation for all target operating points."""
        target_qps = [22, 27, 32, 37, 42, 47]
        for qp_inter in target_qps:
            qp_intra, qp_out = self.injector.calculate_qp_pair(qp_inter)
            self.assertEqual(qp_out, qp_inter)
            self.assertEqual(qp_intra, qp_inter - 5)

        # Boundary clamping
        self.assertEqual(self.injector.calculate_qp_pair(4), (0, 4))
        self.assertEqual(self.injector.calculate_qp_pair(0), (0, 0))

        # Invalid QPs
        with self.assertRaises(ValueError):
            self.injector.calculate_qp_pair(64)
        with self.assertRaises(ValueError):
            self.injector.calculate_qp_pair(-1)

    # -------------------------------------------------------------------------
    # 2. Sequence Construction from NumPy Arrays
    # -------------------------------------------------------------------------
    def test_02_build_injected_sequence_numpy(self):
        """Verify build_injected_sequence correctly places neural I-frame at POC 0."""
        w, h = 64, 64
        # Synthetic Neural I-frame (bright red)
        i_frame_rgb = np.zeros((h, w, 3), dtype=np.float32)
        i_frame_rgb[:, :, 0] = 1.0

        # Synthetic Inter frames (Frame 1: green, Frame 2: blue)
        p1 = np.zeros((h, w, 3), dtype=np.float32)
        p1[:, :, 1] = 1.0
        p2 = np.zeros((h, w, 3), dtype=np.float32)
        p2[:, :, 2] = 1.0

        out_yuv = Path(self.temp_dir) / "test_seq_np.yuv"
        self.injector.build_injected_sequence(
            neural_i_frame=i_frame_rgb,
            inter_frames=[p1, p2],
            width=w,
            height=h,
            output_yuv_path=out_yuv,
            standard="bt709",
            bit_depth=8,
        )

        self.assertTrue(out_yuv.is_file())
        self.assertEqual(get_yuv_frame_count(out_yuv, w, h, bit_depth=8), 3)

        # Frame 0 should match red YUV conversion
        y0_exp, u0_exp, v0_exp = rgb_to_yuv420(i_frame_rgb, standard="bt709", bit_depth=8)
        y0_act, u0_act, v0_act = read_yuv_frame(out_yuv, w, h, frame_idx=0, bit_depth=8)
        np.testing.assert_array_equal(y0_exp, y0_act)
        np.testing.assert_array_equal(u0_exp, u0_act)
        np.testing.assert_array_equal(v0_exp, v0_act)

    # -------------------------------------------------------------------------
    # 3. Sequence Construction from PyTorch Tensors
    # -------------------------------------------------------------------------
    def test_03_build_injected_sequence_tensors(self):
        """Verify build_injected_sequence accepts PyTorch float tensors."""
        w, h = 32, 32
        t_i = torch.rand(3, h, w)
        t_p1 = torch.rand(3, h, w)

        out_yuv = Path(self.temp_dir) / "test_seq_tensor.yuv"
        self.injector.build_injected_sequence(
            neural_i_frame=t_i,
            inter_frames=[t_p1],
            width=w,
            height=h,
            output_yuv_path=out_yuv,
            standard="bt709",
            bit_depth=8,
        )

        self.assertTrue(out_yuv.is_file())
        self.assertEqual(get_yuv_frame_count(out_yuv, w, h, bit_depth=8), 2)

    # -------------------------------------------------------------------------
    # 4. Real VTM 12.0 Low-Delay P Reference Injection Encoding & Decoding
    # -------------------------------------------------------------------------
    def test_04_real_vtm_ldp_reference_injection(self):
        """Verify real VTM 12.0 encoding and decoding using reference-injected sequence in LDP mode."""
        w, h = 64, 64
        # Neural reconstructed I-frame (smooth gradient)
        i_frame = np.linspace(0.1, 0.9, w * h * 3, dtype=np.float32).reshape(h, w, 3)
        # Inter frames (small temporal differences)
        p1 = i_frame * 0.95
        p2 = i_frame * 0.90

        bitstream_path = Path(self.temp_dir) / "ref_inj_test.vvc"
        recon_enc_path = Path(self.temp_dir) / "ref_inj_recon_enc.yuv"
        recon_dec_path = Path(self.temp_dir) / "ref_inj_recon_dec.yuv"

        # Encode with reference injection at target QP 32 (Intra QP 27)
        res = self.injector.encode_with_reference_injection(
            neural_i_frame=i_frame,
            inter_frames=[p1, p2],
            width=w,
            height=h,
            qp_inter=32,
            output_bitstream=bitstream_path,
            recon_yuv=recon_enc_path,
            cfg_name_or_path="encoder_lowdelay_P_vtm.cfg",
            frame_rate=30,
        )

        self.assertIsInstance(res, VTMEncodeResult)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(bitstream_path.is_file())
        self.assertGreater(bitstream_path.stat().st_size, 0)
        self.assertTrue(recon_enc_path.is_file())
        self.assertEqual(res.frames_encoded, 3)
        self.assertIsNotNone(res.total_bits)

        # Decode with VTM 12.0 Decoder
        dec_res = self.vtm.decode(
            bitstream_path=bitstream_path,
            output_recon_path=recon_dec_path,
        )
        self.assertIsInstance(dec_res, VTMDecodeResult)
        self.assertEqual(dec_res.returncode, 0)
        self.assertTrue(recon_dec_path.is_file())

        # Bit-for-bit verification of reconstructed sequence
        with open(recon_enc_path, "rb") as f_enc, open(recon_dec_path, "rb") as f_dec:
            self.assertEqual(f_enc.read(), f_dec.read(), "Encoder and Decoder reconstructions do not match bit-for-bit!")


if __name__ == "__main__":
    unittest.main()

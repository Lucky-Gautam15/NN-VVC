import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Tuple, Union


import numpy as np
import torch

from src.adapters.iha import IntraHumanAdapter
from src.lic.lic_model import LICModel
from src.pipeline.nn_vvc_codec import (
    NNVVCCodec,
    NNVVCDecodeResult,
    NNVVCEncodeResult,
)
from src.vvc.muxer import NNVVCDeMuxer
from src.vvc.vtm_wrapper import VTMWrapper
from src.vvc.yuv_utils import (
    calculate_yuv420_frame_bytes,
    get_yuv_frame_count,
    read_yuv_sequence,
    write_yuv_sequence,
)


class TestPhaseEPipeline(unittest.TestCase):
    """
    Test suite for Phase E-5 End-to-End NN-VVC hybrid video codec orchestrator.
    Validates LIC + IHA + Reference Injection + VTM 12.0 Low-Delay P + Muxer integration.
    """

    @classmethod
    def setUpClass(cls):
        temp_base = Path(r"E:\temp") if Path(r"E:\temp").is_dir() else None
        cls.temp_dir = tempfile.mkdtemp(prefix="nnvvc_pipe_test_", dir=temp_base)
        cls.codec = NNVVCCodec(temp_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def create_synthetic_sequence(self, width: int = 64, height: int = 64, frame_count: int = 3) -> List[np.ndarray]:
        """Generate a deterministic synthetic RGB video sequence in [0.0, 1.0]."""
        frames = []
        for i in range(frame_count):
            r = np.linspace(0.1, 0.9, width).reshape(1, width).repeat(height, axis=0)
            g = np.linspace(0.2, 0.8, height).reshape(height, 1).repeat(width, axis=1)
            b = np.full((height, width), (i + 1) * 0.25)
            frame = np.stack([r, g, b], axis=-1).astype(np.float32)
            frames.append(frame)
        return frames

    # -------------------------------------------------------------------------
    # 1. Codec Initialization & Component Wiring
    # -------------------------------------------------------------------------
    def test_01_codec_initialization(self):
        """Verify codec instantiates LICModel, IHA, VTMWrapper, and ReferenceInjector."""
        self.assertIsInstance(self.codec.lic_model, LICModel)
        self.assertIsInstance(self.codec.iha_model, IntraHumanAdapter)
        self.assertIsInstance(self.codec.vtm, VTMWrapper)
        self.assertTrue(self.codec.temp_dir.is_dir())

    # -------------------------------------------------------------------------
    # 2. Neural Latent Serialization
    # -------------------------------------------------------------------------
    def test_02_neural_latent_serialization_roundtrip(self):
        """Verify losslessly packing and unpacking quantized latent tensors."""
        latent = torch.randn(1, 192, 4, 4)
        raw_bytes = NNVVCCodec.serialize_neural_latent(latent)
        self.assertIsInstance(raw_bytes, bytes)
        self.assertGreater(len(raw_bytes), 20)

        restored_latent = NNVVCCodec.deserialize_neural_latent(raw_bytes)
        self.assertEqual(restored_latent.shape, latent.shape)
        self.assertTrue(torch.allclose(latent, restored_latent, atol=1e-6))

    # -------------------------------------------------------------------------
    # 3. Resolution Scaling Helpers
    # -------------------------------------------------------------------------
    def test_03_resolution_scaling_helpers(self):
        """Verify 3/4 downsampling and 4/3 upsampling dimensional consistency."""
        t_in = torch.rand(1, 3, 64, 64)
        scaled_t, orig_dim = NNVVCCodec.downscale_3_4(t_in)
        self.assertEqual(orig_dim, (64, 64))
        self.assertEqual(scaled_t.shape, (1, 3, 48, 48))

        upscaled_t = NNVVCCodec.upscale_4_3(scaled_t, orig_dim)
        self.assertEqual(upscaled_t.shape, (1, 3, 64, 64))

    # -------------------------------------------------------------------------
    # 4. End-to-End Encoding & .nnvvc Generation
    # -------------------------------------------------------------------------
    def test_04_e2e_encode_sequence(self):
        """Verify end-to-end encoding of synthetic sequence into .nnvvc container."""
        w, h, n_frames = 64, 64, 3
        frames = self.create_synthetic_sequence(w, h, n_frames)

        out_nnvvc = Path(self.temp_dir) / "test_04.nnvvc"
        out_recon = Path(self.temp_dir) / "test_04_enc_recon.yuv"

        result = self.codec.encode_sequence(
            input_sequence=frames,
            width=w,
            height=h,
            qp_inter=32,
            output_bitstream=out_nnvvc,
            recon_yuv=out_recon,
        )

        self.assertIsInstance(result, NNVVCEncodeResult)
        self.assertTrue(out_nnvvc.is_file())
        self.assertGreater(out_nnvvc.stat().st_size, 0)
        self.assertEqual(result.frames_encoded, n_frames)
        self.assertGreater(result.neural_bits, 0)
        self.assertGreater(result.vtm_bits, 0)
        self.assertGreater(result.total_bits, result.neural_bits + result.vtm_bits)
        self.assertIsNotNone(result.psnr_y)
        self.assertTrue(out_recon.is_file())
        self.assertEqual(out_recon.stat().st_size, calculate_yuv420_frame_bytes(w, h) * n_frames)

        # Inspect container header
        hdr = NNVVCDeMuxer.read_header(out_nnvvc)
        self.assertEqual(hdr.width, w)
        self.assertEqual(hdr.height, h)
        self.assertEqual(hdr.frame_count, n_frames)
        self.assertEqual(hdr.qp_inter, 32)
        self.assertEqual(hdr.qp_intra, 27)  # QP_intra = QP_inter - 5

    # -------------------------------------------------------------------------
    # 5. End-to-End Decoding & Full Reconstruction
    # -------------------------------------------------------------------------
    def test_05_e2e_decode_sequence(self):
        """Verify decoding .nnvvc bitstream into reconstructed YUV video sequence."""
        w, h, n_frames = 64, 64, 2
        frames = self.create_synthetic_sequence(w, h, n_frames)

        out_nnvvc = Path(self.temp_dir) / "test_05.nnvvc"
        out_recon = Path(self.temp_dir) / "test_05_dec_recon.yuv"

        self.codec.encode_sequence(
            input_sequence=frames,
            width=w,
            height=h,
            qp_inter=27,
            output_bitstream=out_nnvvc,
        )

        dec_result = self.codec.decode_sequence(
            bitstream_path=out_nnvvc,
            output_recon_path=out_recon,
            return_frames=True,
        )

        self.assertIsInstance(dec_result, NNVVCDecodeResult)
        self.assertEqual(dec_result.frames_decoded, n_frames)
        self.assertEqual(dec_result.width, w)
        self.assertEqual(dec_result.height, h)
        self.assertTrue(out_recon.is_file())
        self.assertEqual(out_recon.stat().st_size, calculate_yuv420_frame_bytes(w, h) * n_frames)
        self.assertIsNotNone(dec_result.reconstructed_frames)
        self.assertEqual(len(dec_result.reconstructed_frames), n_frames)

    # -------------------------------------------------------------------------
    # 6. YUV File Input Path
    # -------------------------------------------------------------------------
    def test_06_yuv_file_input_pipeline(self):
        """Verify pipeline accepts raw .yuv file paths as input."""
        w, h, n_frames = 64, 64, 3
        frames_yuv = []
        for i in range(n_frames):
            y = np.full((h, w), 120 + i * 20, dtype=np.uint8)
            u = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            v = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            frames_yuv.append((y, u, v))

        src_yuv = Path(self.temp_dir) / "test_06_src.yuv"
        write_yuv_sequence(src_yuv, frames_yuv, w, h, bit_depth=8)

        out_nnvvc = Path(self.temp_dir) / "test_06.nnvvc"
        out_recon = Path(self.temp_dir) / "test_06_recon.yuv"

        enc_res = self.codec.encode_sequence(
            input_sequence=src_yuv,
            width=w,
            height=h,
            qp_inter=37,
            output_bitstream=out_nnvvc,
        )
        self.assertEqual(enc_res.frames_encoded, n_frames)

        dec_res = self.codec.decode_sequence(
            bitstream_path=out_nnvvc,
            output_recon_path=out_recon,
        )
        self.assertEqual(dec_res.frames_decoded, n_frames)
        self.assertEqual(get_yuv_frame_count(out_recon, w, h), n_frames)

    # -------------------------------------------------------------------------
    # 7. Resolution Scaling Trigger Pipeline
    # -------------------------------------------------------------------------
    def test_07_resolution_scaling_pipeline(self):
        """Verify pipeline with 3/4 downscale and 4/3 upscale when above resolution threshold."""
        w, h, n_frames = 64, 64, 2
        # Set threshold to 64 so 64x64 triggers scaling
        scaled_codec = NNVVCCodec(temp_dir=self.temp_dir, res_scale_threshold=64)
        frames = self.create_synthetic_sequence(w, h, n_frames)

        out_nnvvc = Path(self.temp_dir) / "test_07_scaled.nnvvc"
        out_recon = Path(self.temp_dir) / "test_07_scaled_recon.yuv"

        enc_res = scaled_codec.encode_sequence(
            input_sequence=frames,
            width=w,
            height=h,
            qp_inter=32,
            output_bitstream=out_nnvvc,
        )
        self.assertEqual(enc_res.frames_encoded, n_frames)

        hdr = NNVVCDeMuxer.read_header(out_nnvvc)
        self.assertEqual(hdr.scale_num, 3)
        self.assertEqual(hdr.scale_denom, 4)

        dec_res = scaled_codec.decode_sequence(
            bitstream_path=out_nnvvc,
            output_recon_path=out_recon,
        )
        self.assertEqual(dec_res.frames_decoded, n_frames)
        self.assertEqual(dec_res.width, w)
        self.assertEqual(dec_res.height, h)


if __name__ == "__main__":
    unittest.main()

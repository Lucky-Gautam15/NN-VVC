import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path


import numpy as np

from src.vvc.muxer import (
    NNVVCContainerHeader,
    NNVVCCorruptedDataError,
    NNVVCDeMuxer,
    NNVVCFormatError,
    NNVVCInvalidMagicError,
    NNVVCMuxer,
    NNVVCPayload,
    NNVVCTruncatedFileError,
    NNVVCUnsupportedVersionError,
)
from src.vvc.vtm_wrapper import VTMWrapper
from src.vvc.yuv_utils import write_yuv_sequence


class TestNNVVCMuxer(unittest.TestCase):
    """
    Test suite for Phase E-4 Hybrid NN-VVC bitstream muxer/demuxer.
    Validates binary container serialization, metadata preservation,
    payload integrity, corruption/truncation detection, and real VTM bitstream roundtrips.
    """

    @classmethod
    def setUpClass(cls):
        temp_base = Path(r"E:\temp") if Path(r"E:\temp").is_dir() else None
        cls.temp_dir = tempfile.mkdtemp(prefix="nnvvc_muxer_test_", dir=temp_base)
        cls.vtm = VTMWrapper()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def create_dummy_header(
        self,
        width: int = 1920,
        height: int = 1080,
        frame_count: int = 30,
        qp_intra: int = 27,
        qp_inter: int = 32,
    ) -> NNVVCContainerHeader:
        return NNVVCContainerHeader(
            version=1,
            width=width,
            height=height,
            frame_count=frame_count,
            framerate=30,
            bit_depth=8,
            chroma_format=420,
            qp_intra=qp_intra,
            qp_inter=qp_inter,
            scale_num=3,
            scale_denom=4,
            extra_metadata=b'{"codec": "LIC-IHA", "target": "machine+human"}',
        )

    # -------------------------------------------------------------------------
    # 1. Header Serialization & Deserialization
    # -------------------------------------------------------------------------
    def test_01_header_serialization_roundtrip(self):
        """Verify NNVVCContainerHeader serializes and deserializes with exact field preservation."""
        hdr_in = self.create_dummy_header(width=1280, height=720, frame_count=60, qp_intra=22, qp_inter=27)
        hdr_in.neural_payload_size = 1024
        hdr_in.vtm_payload_size = 2048
        hdr_in.payload_crc32 = 0x12345678

        serialized = hdr_in.serialize()
        self.assertIsInstance(serialized, bytes)
        self.assertGreaterEqual(len(serialized), NNVVCContainerHeader.FIXED_HEADER_SIZE)

        hdr_out, bytes_consumed = NNVVCContainerHeader.deserialize(serialized)
        self.assertEqual(bytes_consumed, len(serialized))
        self.assertEqual(hdr_out.version, hdr_in.version)
        self.assertEqual(hdr_out.width, hdr_in.width)
        self.assertEqual(hdr_out.height, hdr_in.height)
        self.assertEqual(hdr_out.frame_count, hdr_in.frame_count)
        self.assertEqual(hdr_out.framerate, hdr_in.framerate)
        self.assertEqual(hdr_out.bit_depth, hdr_in.bit_depth)
        self.assertEqual(hdr_out.chroma_format, hdr_in.chroma_format)
        self.assertEqual(hdr_out.qp_intra, hdr_in.qp_intra)
        self.assertEqual(hdr_out.qp_inter, hdr_in.qp_inter)
        self.assertEqual(hdr_out.scale_num, hdr_in.scale_num)
        self.assertEqual(hdr_out.scale_denom, hdr_in.scale_denom)
        self.assertEqual(hdr_out.neural_payload_size, hdr_in.neural_payload_size)
        self.assertEqual(hdr_out.vtm_payload_size, hdr_in.vtm_payload_size)
        self.assertEqual(hdr_out.payload_crc32, hdr_in.payload_crc32)
        self.assertEqual(hdr_out.extra_metadata, hdr_in.extra_metadata)

    # -------------------------------------------------------------------------
    # 2. Magic & Version Validation
    # -------------------------------------------------------------------------
    def test_02_magic_and_version_validation(self):
        """Verify invalid magic identifier and unsupported versions raise appropriate errors."""
        hdr = self.create_dummy_header(width=64, height=64)
        raw = bytearray(hdr.serialize())

        # Corrupt magic
        raw[:5] = b"XXXXX"
        with self.assertRaises(NNVVCInvalidMagicError):
            NNVVCContainerHeader.deserialize(bytes(raw))

        # Unsupported version
        raw = bytearray(hdr.serialize())
        raw[5:7] = struct.pack("!H", 999)  # version 999
        with self.assertRaises(NNVVCUnsupportedVersionError):
            NNVVCContainerHeader.deserialize(bytes(raw))

    # -------------------------------------------------------------------------
    # 3. Payload Preservation & Complete Mux/Demux Roundtrip
    # -------------------------------------------------------------------------
    def test_03_mux_demux_roundtrip_bytes(self):
        """Verify byte-exact preservation of neural and VTM payloads via in-memory mux/demux."""
        hdr = self.create_dummy_header(width=64, height=64, frame_count=2)
        neural_data = b"NEURAL_LATENTS_AND_HYPERPRIOR_DATA_12345" * 10
        vtm_data = b"\x00\x00\x00\x01VTM_NALU_PAYLOAD_CHUNK_67890" * 20

        container_bytes = NNVVCMuxer.mux(hdr, neural_data, vtm_data)
        self.assertIsInstance(container_bytes, bytes)

        payload = NNVVCDeMuxer.demux(container_bytes, verify_checksum=True)
        self.assertEqual(payload.neural_payload, neural_data)
        self.assertEqual(payload.vtm_payload, vtm_data)
        self.assertEqual(payload.header.width, 64)
        self.assertEqual(payload.header.height, 64)
        self.assertEqual(payload.header.neural_payload_size, len(neural_data))
        self.assertEqual(payload.header.vtm_payload_size, len(vtm_data))

    # -------------------------------------------------------------------------
    # 4. File-Based Mux and Demux
    # -------------------------------------------------------------------------
    def test_04_mux_demux_file_io(self):
        """Verify file-based multiplexing and demultiplexing."""
        hdr = self.create_dummy_header(width=128, height=128, frame_count=4)
        neural_data = os.urandom(512)
        vtm_data = os.urandom(2048)

        file_path = Path(self.temp_dir) / "test_container.nnvvc"
        NNVVCMuxer.mux(hdr, neural_data, vtm_data, output_path_or_handle=file_path)

        self.assertTrue(file_path.is_file())
        self.assertGreater(file_path.stat().st_size, len(neural_data) + len(vtm_data))

        # Fast header extraction without loading whole file
        extracted_hdr = NNVVCDeMuxer.read_header(file_path)
        self.assertEqual(extracted_hdr.width, 128)
        self.assertEqual(extracted_hdr.height, 128)

        # Full demux
        payload = NNVVCDeMuxer.demux(file_path, verify_checksum=True)
        self.assertEqual(payload.neural_payload, neural_data)
        self.assertEqual(payload.vtm_payload, vtm_data)

    # -------------------------------------------------------------------------
    # 5. Empty/Zero-Length Payload Handling
    # -------------------------------------------------------------------------
    def test_05_empty_payloads_handling(self):
        """Verify container handles zero-length neural or VTM payloads cleanly."""
        hdr = self.create_dummy_header(width=64, height=64, frame_count=1)
        neural_data = b""
        vtm_data = b"ONLY_VTM_PAYLOAD"

        container_bytes = NNVVCMuxer.mux(hdr, neural_data, vtm_data)
        payload = NNVVCDeMuxer.demux(container_bytes, verify_checksum=True)
        self.assertEqual(payload.neural_payload, b"")
        self.assertEqual(payload.vtm_payload, vtm_data)

    # -------------------------------------------------------------------------
    # 6. Large Payload Handling
    # -------------------------------------------------------------------------
    def test_06_large_payload_handling(self):
        """Verify container handles multi-megabyte payloads correctly."""
        hdr = self.create_dummy_header(width=1920, height=1080, frame_count=120)
        neural_data = os.urandom(1024 * 1024)      # 1 MB
        vtm_data = os.urandom(2 * 1024 * 1024)    # 2 MB

        container_bytes = NNVVCMuxer.mux(hdr, neural_data, vtm_data)
        payload = NNVVCDeMuxer.demux(container_bytes, verify_checksum=True)
        self.assertEqual(len(payload.neural_payload), 1024 * 1024)
        self.assertEqual(len(payload.vtm_payload), 2 * 1024 * 1024)

    # -------------------------------------------------------------------------
    # 7. Truncation Detection
    # -------------------------------------------------------------------------
    def test_07_truncation_detection(self):
        """Verify truncated files raise NNVVCTruncatedFileError."""
        hdr = self.create_dummy_header(width=64, height=64)
        neural_data = b"SOME_NEURAL_BITS"
        vtm_data = b"SOME_VTM_BITS"

        full_container = NNVVCMuxer.mux(hdr, neural_data, vtm_data)

        # Truncate at header level
        with self.assertRaises(NNVVCTruncatedFileError):
            NNVVCDeMuxer.demux(full_container[:20])

        # Truncate inside payload
        with self.assertRaises(NNVVCTruncatedFileError):
            NNVVCDeMuxer.demux(full_container[:-5])

    # -------------------------------------------------------------------------
    # 8. Checksum Corruption Detection
    # -------------------------------------------------------------------------
    def test_08_corruption_detection(self):
        """Verify byte corruption in payload triggers NNVVCCorruptedDataError."""
        hdr = self.create_dummy_header(width=64, height=64)
        neural_data = b"CRITICAL_NEURAL_BYTES_1234"
        vtm_data = b"CRITICAL_VTM_BYTES_5678"

        container_bytes = bytearray(NNVVCMuxer.mux(hdr, neural_data, vtm_data))

        # Corrupt 1 byte in neural payload
        corrupt_idx = len(container_bytes) - 5
        container_bytes[corrupt_idx] ^= 0xFF

        with self.assertRaises(NNVVCCorruptedDataError):
            NNVVCDeMuxer.demux(bytes(container_bytes), verify_checksum=True)

    # -------------------------------------------------------------------------
    # 9. Real VTM 12.0 .vvc Bitstream Multiplexing Roundtrip
    # -------------------------------------------------------------------------
    def test_09_real_vtm_bitstream_mux_demux_roundtrip(self):
        """Verify muxing a real VTM 12.0 encoded .vvc bitstream and decoding it after demuxing."""
        w, h, n_frames = 64, 64, 2
        # Generate synthetic sequence
        frames = []
        for i in range(n_frames):
            y = np.full((h, w), 100 + i * 40, dtype=np.uint8)
            u = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            v = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            frames.append((y, u, v))

        src_yuv = Path(self.temp_dir) / "vtm_mux_in.yuv"
        vtm_bitstream_path = Path(self.temp_dir) / "vtm_mux_orig.vvc"
        write_yuv_sequence(src_yuv, frames, w, h, bit_depth=8)

        # 1. Encode with real VTM 12.0
        self.vtm.encode(
            input_yuv=src_yuv,
            width=w,
            height=h,
            frame_count=n_frames,
            qp=32,
            output_bitstream=vtm_bitstream_path,
        )
        with open(vtm_bitstream_path, "rb") as f:
            vtm_raw_bytes = f.read()

        # 2. Package into .nnvvc container with synthetic neural latent payload
        neural_latent_bytes = b"MOCK_LIC_LATENT_WEIGHTS_AND_PROBABILITIES"
        hdr = NNVVCContainerHeader(
            width=w,
            height=h,
            frame_count=n_frames,
            framerate=30,
            bit_depth=8,
            chroma_format=420,
            qp_intra=27,
            qp_inter=32,
        )
        container_path = Path(self.temp_dir) / "packaged_video.nnvvc"
        NNVVCMuxer.mux(hdr, neural_latent_bytes, vtm_raw_bytes, output_path_or_handle=container_path)

        # 3. Demux from .nnvvc file
        payload = NNVVCDeMuxer.demux(container_path, verify_checksum=True)
        self.assertEqual(payload.neural_payload, neural_latent_bytes)
        self.assertEqual(payload.vtm_payload, vtm_raw_bytes)

        # 4. Decode the extracted VTM payload with real VTM 12.0 DecoderApp
        extracted_vvc_path = Path(self.temp_dir) / "extracted.vvc"
        recon_demux_path = Path(self.temp_dir) / "recon_demux.yuv"
        with open(extracted_vvc_path, "wb") as f:
            f.write(payload.vtm_payload)

        dec_result = self.vtm.decode(
            bitstream_path=extracted_vvc_path,
            output_recon_path=recon_demux_path,
        )
        self.assertEqual(dec_result.returncode, 0)
        self.assertTrue(recon_demux_path.is_file())


if __name__ == "__main__":
    unittest.main()

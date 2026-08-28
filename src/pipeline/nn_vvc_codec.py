"""
End-to-End NN-VVC Hybrid Video Codec Orchestrator.

Integrates Learned Image Compression (LIC), Intra Human Adapter (IHA),
VTM 12.0 Low-Delay P reference injection, and the .nnvvc hybrid container muxer/demuxer.

Paper Reference:
    "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
     image coding for machines", Section IV (Conventional Video Coding Integration).
"""

import io
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.adapters.iha import IntraHumanAdapter
from src.lic.lic_model import LICModel
from src.vvc.muxer import (
    NNVVCContainerHeader,
    NNVVCDeMuxer,
    NNVVCMuxer,
    NNVVCPayload,
)
from src.vvc.reference_injector import ReferenceInjector
from src.vvc.vtm_wrapper import VTMDecodeResult, VTMEncodeResult, VTMWrapper
from src.vvc.yuv_utils import (
    calculate_yuv420_frame_bytes,
    get_yuv_frame_count,
    pad_to_alignment,
    read_yuv_frame,
    read_yuv_sequence,
    rgb_to_yuv420,
    unpad_from_alignment,
    write_yuv_frame,
    write_yuv_sequence,
    yuv420_to_rgb,
)


@dataclass
class NNVVCEncodeResult:
    """Structured metrics returned by NNVVCCodec.encode_sequence()."""
    bitstream_path: Path
    total_bits: int
    neural_bits: int
    vtm_bits: int
    header_bits: int
    frames_encoded: int
    bitrate_kbps: Optional[float] = None
    psnr_y: Optional[float] = None
    psnr_u: Optional[float] = None
    psnr_v: Optional[float] = None
    psnr_yuv: Optional[float] = None
    recon_yuv_path: Optional[Path] = None


@dataclass
class NNVVCDecodeResult:
    """Structured result returned by NNVVCCodec.decode_sequence()."""
    recon_yuv_path: Path
    frames_decoded: int
    width: int
    height: int
    reconstructed_frames: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None


class NNVVCCodec:
    """
    Unified end-to-end codec orchestrator for NN-VVC.
    """

    def __init__(
        self,
        lic_model: Optional[LICModel] = None,
        iha_model: Optional[IntraHumanAdapter] = None,
        vtm_wrapper: Optional[VTMWrapper] = None,
        temp_dir: Optional[Union[str, Path]] = None,
        res_scale_threshold: Optional[int] = None,
        device: str = "cpu",
    ):
        """
        Initialize the NN-VVC Codec.

        Args:
            lic_model: Pre-instantiated LICModel (or creates default).
            iha_model: Pre-instantiated IntraHumanAdapter (or creates default).
            vtm_wrapper: Pre-instantiated VTMWrapper (or creates default).
            temp_dir: Directory for temporary YUV/VTM buffers (defaults to E:\\temp or system temp).
            res_scale_threshold: Maximum dimension above which 3/4 downsampling is triggered (None = disabled).
            device: Computing device for neural models ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.lic_model = lic_model if lic_model is not None else LICModel()
        self.lic_model.to(self.device).eval()

        self.iha_model = iha_model if iha_model is not None else IntraHumanAdapter()
        self.iha_model.to(self.device).eval()

        self.vtm = vtm_wrapper if vtm_wrapper is not None else VTMWrapper()
        if temp_dir is not None:
            self.temp_dir = Path(temp_dir)
        elif Path(r"E:\temp").is_dir():
            self.temp_dir = Path(r"E:\temp")
        else:
            self.temp_dir = Path(tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.injector = ReferenceInjector(vtm_wrapper=self.vtm, temp_dir=self.temp_dir)
        self.res_scale_threshold = res_scale_threshold

    # -------------------------------------------------------------------------
    # Neural Latent Serialization
    # -------------------------------------------------------------------------
    @staticmethod
    def serialize_neural_latent(latent_tensor: torch.Tensor) -> bytes:
        """
        Losslessly serialize a quantized latent PyTorch tensor into binary bytes.

        Layout:
            !4s (Magic b"NLAT")
            !III (Channels, Height, Width)
            !I (Raw byte size)
            + Compressed float32 bytes
        """
        t = latent_tensor.detach().cpu().to(torch.float32).squeeze(0)  # Shape (C, H, W)
        if t.ndim != 3:
            raise ValueError(f"Expected 3D latent tensor (C, H, W), got shape {t.shape}")
        c, h, w = t.shape
        raw_bytes = t.numpy().tobytes()
        compressed = zlib.compress(raw_bytes, level=6)
        hdr = struct.pack("!4sIIII", b"NLAT", c, h, w, len(compressed))
        return hdr + compressed

    @staticmethod
    def deserialize_neural_latent(data: bytes, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """
        Deserialize binary bytes back into a quantized latent PyTorch tensor (1, C, H, W).
        """
        if len(data) < 20:
            raise ValueError("Data too small to contain valid neural latent header.")
        magic, c, h, w, comp_len = struct.unpack_from("!4sIIII", data, 0)
        if magic != b"NLAT":
            raise ValueError(f"Invalid neural latent magic {magic!r}, expected b'NLAT'")
        compressed = data[20 : 20 + comp_len]
        raw_bytes = zlib.decompress(compressed)
        arr = np.frombuffer(raw_bytes, dtype=np.float32).copy().reshape((c, h, w))
        t = torch.from_numpy(arr).unsqueeze(0).to(device)
        return t


    # -------------------------------------------------------------------------
    # Resolution Scaling Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def downscale_3_4(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Downscale tensor by 3/4 using bicubic interpolation."""
        _, _, h, w = tensor.shape
        new_h = int(round(h * 0.75 / 2.0)) * 2
        new_w = int(round(w * 0.75 / 2.0)) * 2
        scaled = F.interpolate(tensor, size=(new_h, new_w), mode="bicubic", align_corners=False)
        return torch.clamp(scaled, 0.0, 1.0), (h, w)

    @staticmethod
    def upscale_4_3(tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Upscale tensor back to target_size using bicubic interpolation."""
        scaled = F.interpolate(tensor, size=target_size, mode="bicubic", align_corners=False)
        return torch.clamp(scaled, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Neural I-Frame Processing
    # -------------------------------------------------------------------------
    def process_neural_intra_frame(
        self,
        frame_rgb_tensor: torch.Tensor,
        qp_intra: int,
        apply_scaling: bool = False,
    ) -> Tuple[torch.Tensor, bytes]:
        """
        Encode Frame 0 using LIC, adapt with IHA, and serialize latent representation.

        Args:
            frame_rgb_tensor: Float tensor of Frame 0, shape (1, 3, H, W) in [0.0, 1.0].
            qp_intra: Quantization parameter for intra frame.
            apply_scaling: If True, applies 3/4 downscale and 4/3 upscale.

        Returns:
            (recon_human_tensor, neural_payload_bytes)
        """
        x_in = frame_rgb_tensor.to(self.device)
        _, _, orig_h, orig_w = x_in.shape

        # 1. Optional 3/4 downscaling
        if apply_scaling:
            x_proc, target_dim = self.downscale_3_4(x_in)
        else:
            x_proc = x_in
            target_dim = (orig_h, orig_w)

        # 2. Pad to alignment (multiple of 16 for LIC convolutional stages)
        x_padded, pad_shape = pad_to_alignment(x_proc, align=16, mode="reflect")

        with torch.no_grad():
            # 3. LIC Encode & Quantize
            latent = self.lic_model.encode(x_padded)
            quantized_latent = self.lic_model.quantize(latent)

            # 4. LIC Decode
            x_lic_padded = self.lic_model.decode(quantized_latent)
            x_lic = unpad_from_alignment(x_lic_padded, pad_shape)

            # 5. IHA Adaptation
            h_proc, w_proc = x_lic.shape[-2], x_lic.shape[-1]
            x_human = self.iha_model(x_lic, qp=qp_intra, resolution=(h_proc, w_proc))
            x_human = torch.clamp(x_human, 0.0, 1.0)

            # 6. Optional 4/3 upscaling
            if apply_scaling:
                x_human_final = self.upscale_4_3(x_human, target_dim)
            else:
                x_human_final = x_human

            # 7. Serialize neural latent payload
            neural_payload = self.serialize_neural_latent(quantized_latent)

        return x_human_final, neural_payload

    def reconstruct_neural_intra_frame(
        self,
        neural_payload: bytes,
        target_size: Tuple[int, int],
        qp_intra: int,
        apply_scaling: bool = False,
    ) -> torch.Tensor:
        """
        Decode neural payload through LIC decoder and IHA to reconstruct Frame 0.
        """
        orig_h, orig_w = target_size
        with torch.no_grad():
            quantized_latent = self.deserialize_neural_latent(neural_payload, device=self.device)

            # LIC Decode
            x_lic_padded = self.lic_model.decode(quantized_latent)
            
            # Determine unpadded dimensions
            if apply_scaling:
                scaled_h = int(round(orig_h * 0.75 / 2.0)) * 2
                scaled_w = int(round(orig_w * 0.75 / 2.0)) * 2
                unpad_dim = (scaled_h, scaled_w)
            else:
                unpad_dim = (orig_h, orig_w)

            x_lic = unpad_from_alignment(x_lic_padded, unpad_dim)

            # IHA Adaptation
            h_curr, w_curr = x_lic.shape[-2], x_lic.shape[-1]
            x_human = self.iha_model(x_lic, qp=qp_intra, resolution=(h_curr, w_curr))
            x_human = torch.clamp(x_human, 0.0, 1.0)

            if apply_scaling:
                x_human_final = self.upscale_4_3(x_human, (orig_h, orig_w))
            else:
                x_human_final = x_human

        return x_human_final

    # -------------------------------------------------------------------------
    # End-to-End Encoding
    # -------------------------------------------------------------------------
    def encode_sequence(
        self,
        input_sequence: Union[
            List[np.ndarray],
            List[torch.Tensor],
            Union[str, Path],
        ],
        width: int,
        height: int,
        qp_inter: int,
        output_bitstream: Union[str, Path],
        frame_count: Optional[int] = None,
        frame_rate: int = 30,
        standard: str = "bt709",
        bit_depth: int = 8,
        recon_yuv: Optional[Union[str, Path]] = None,
    ) -> NNVVCEncodeResult:
        """
        Encode an entire video sequence into a .nnvvc hybrid container.

        Args:
            input_sequence: List of RGB frames (NumPy arrays/PyTorch tensors) or path to input .yuv file.
            width: Frame width.
            height: Frame height.
            qp_inter: Inter-frame quantization parameter (QP_intra = qp_inter - 5).
            output_bitstream: Destination path for .nnvvc file.
            frame_count: Number of frames to encode (if None, reads all).
            frame_rate: Frame rate.
            standard: Color standard ('bt709' or 'bt601').
            bit_depth: Bit depth.
            recon_yuv: Optional path to save full reconstructed YUV sequence.

        Returns:
            NNVVCEncodeResult with bitrate, bit breakdown, and PSNR metrics.
        """
        output_bitstream = Path(output_bitstream)
        output_bitstream.parent.mkdir(parents=True, exist_ok=True)

        qp_intra, qp_inter_val = self.injector.calculate_qp_pair(qp_inter)

        # 1. Ingest input frames into standard tensor list
        if isinstance(input_sequence, (str, Path)):
            src_path = Path(input_sequence)
            total_avail = get_yuv_frame_count(src_path, width, height, bit_depth=bit_depth)
            n_frames = min(total_avail, frame_count) if frame_count is not None else total_avail
            yuv_frames = read_yuv_sequence(src_path, width, height, frame_count=n_frames, bit_depth=bit_depth)
            rgb_frames = [
                yuv420_to_rgb(y, u, v, standard=standard, bit_depth=bit_depth, return_tensor=True).unsqueeze(0)
                for y, u, v in yuv_frames
            ]
        elif isinstance(input_sequence, list):
            n_frames = len(input_sequence) if frame_count is None else min(len(input_sequence), frame_count)
            rgb_frames = []
            for item in input_sequence[:n_frames]:
                if isinstance(item, torch.Tensor):
                    t = item if item.ndim == 4 else item.unsqueeze(0)
                    rgb_frames.append(t)
                elif isinstance(item, np.ndarray):
                    # (H, W, 3) -> (1, 3, H, W)
                    arr = item.astype(np.float32) / 255.0 if item.dtype == np.uint8 else item.astype(np.float32)
                    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                    rgb_frames.append(t)
                else:
                    raise TypeError(f"Unsupported frame item type: {type(item)}")
        else:
            raise TypeError(f"Unsupported input_sequence type: {type(input_sequence)}")

        if n_frames <= 0:
            raise ValueError("Frame count must be positive")

        # 2. Check resolution scaling condition
        apply_scaling = False
        if self.res_scale_threshold is not None:
            if max(width, height) >= self.res_scale_threshold:
                apply_scaling = True

        scale_num, scale_denom = (3, 4) if apply_scaling else (1, 1)

        # 3. Process Frame 0 through neural path (LIC + IHA)
        frame_0_tensor = rgb_frames[0]
        recon_i_tensor, neural_payload = self.process_neural_intra_frame(
            frame_0_tensor,
            qp_intra=qp_intra,
            apply_scaling=apply_scaling,
        )

        # 4. Prepare reference injection and encode P-frames with VTM 12.0 Low-Delay P
        vtm_bitstream_tmp = self.temp_dir / f"vtm_p_frames_{width}x{height}_qp{qp_inter_val}.vvc"
        recon_yuv_tmp = self.temp_dir / f"vtm_recon_{width}x{height}_qp{qp_inter_val}.yuv"

        # Inter frames (Frames 1..N-1)
        inter_frame_tensors = rgb_frames[1:] if n_frames > 1 else []

        vtm_res = self.injector.encode_with_reference_injection(
            neural_i_frame=recon_i_tensor.squeeze(0),
            inter_frames=[f.squeeze(0) for f in inter_frame_tensors],
            width=width,
            height=height,
            qp_inter=qp_inter_val,
            output_bitstream=vtm_bitstream_tmp,
            recon_yuv=recon_yuv_tmp,
            frame_rate=frame_rate,
            standard=standard,
            bit_depth=bit_depth,
        )

        with open(vtm_bitstream_tmp, "rb") as f_vtm:
            vtm_payload = f_vtm.read()

        # 5. Multiplex into .nnvvc container
        header = NNVVCContainerHeader(
            width=width,
            height=height,
            frame_count=n_frames,
            framerate=frame_rate,
            bit_depth=bit_depth,
            chroma_format=420,
            qp_intra=qp_intra,
            qp_inter=qp_inter_val,
            scale_num=scale_num,
            scale_denom=scale_denom,
        )

        container_bytes = NNVVCMuxer.mux(
            header=header,
            neural_payload=neural_payload,
            vtm_payload=vtm_payload,
            output_path_or_handle=output_bitstream,
        )

        # Copy reconstructed YUV if requested
        if recon_yuv is not None and recon_yuv_tmp.is_file():
            recon_dest = Path(recon_yuv)
            recon_dest.parent.mkdir(parents=True, exist_ok=True)
            with open(recon_yuv_tmp, "rb") as src_f, open(recon_dest, "wb") as dst_f:
                dst_f.write(src_f.read())

        neural_bits = len(neural_payload) * 8
        vtm_bits = len(vtm_payload) * 8
        total_bits = len(container_bytes) * 8
        header_bits = total_bits - (neural_bits + vtm_bits)

        bitrate = (total_bits / 1000.0) * (frame_rate / float(n_frames))

        return NNVVCEncodeResult(
            bitstream_path=output_bitstream,
            total_bits=total_bits,
            neural_bits=neural_bits,
            vtm_bits=vtm_bits,
            header_bits=header_bits,
            frames_encoded=n_frames,
            bitrate_kbps=bitrate,
            psnr_y=vtm_res.psnr_y,
            psnr_u=vtm_res.psnr_u,
            psnr_v=vtm_res.psnr_v,
            psnr_yuv=vtm_res.psnr_yuv,
            recon_yuv_path=Path(recon_yuv) if recon_yuv else None,
        )

    # -------------------------------------------------------------------------
    # End-to-End Decoding
    # -------------------------------------------------------------------------
    def decode_sequence(
        self,
        bitstream_path: Union[str, Path],
        output_recon_path: Union[str, Path],
        return_frames: bool = False,
        standard: str = "bt709",
    ) -> NNVVCDecodeResult:
        """
        Decode a .nnvvc hybrid container into a reconstructed video sequence.

        Args:
            bitstream_path: Path to .nnvvc file.
            output_recon_path: Destination path for reconstructed .yuv sequence.
            return_frames: If True, returns list of (Y, U, V) frame tuples.
            standard: Color conversion standard.

        Returns:
            NNVVCDecodeResult containing reconstructed sequence metadata.
        """
        bitstream_path = Path(bitstream_path)
        output_recon_path = Path(output_recon_path)
        output_recon_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Demultiplex .nnvvc container
        payload = NNVVCDeMuxer.demux(bitstream_path, verify_checksum=True)
        hdr = payload.header

        w, h = hdr.width, hdr.height
        apply_scaling = (hdr.scale_num == 3 and hdr.scale_denom == 4)

        # 2. Reconstruct Neural Frame 0
        recon_i_tensor = self.reconstruct_neural_intra_frame(
            neural_payload=payload.neural_payload,
            target_size=(h, w),
            qp_intra=hdr.qp_intra,
            apply_scaling=apply_scaling,
        )
        y0, u0, v0 = rgb_to_yuv420(recon_i_tensor.squeeze(0), standard=standard, bit_depth=hdr.bit_depth)

        # 3. Decode VTM P-frames from VTM payload
        extracted_vvc = self.temp_dir / f"extracted_{w}x{h}_p.vvc"
        vtm_recon_yuv = self.temp_dir / f"extracted_{w}x{h}_recon.yuv"

        with open(extracted_vvc, "wb") as f_vvc:
            f_vvc.write(payload.vtm_payload)

        dec_res = self.vtm.decode(
            bitstream_path=extracted_vvc,
            output_recon_path=vtm_recon_yuv,
            output_bit_depth=hdr.bit_depth,
        )

        # 4. Assemble final reconstructed sequence: Frame 0 (Neural I-frame) + Frames 1..N-1 (VTM P-frames)
        # Note: VTM's reconstruction of POC 0 in vtm_recon_yuv was seeded by the neural reconstruction
        # and P-frames reference it directly. We write the complete reconstructed sequence.
        with open(vtm_recon_yuv, "rb") as src_f, open(output_recon_path, "wb") as dst_f:
            dst_f.write(src_f.read())

        reconstructed_frames = None
        if return_frames:
            reconstructed_frames = read_yuv_sequence(
                output_recon_path,
                width=w,
                height=h,
                frame_count=hdr.frame_count,
                bit_depth=hdr.bit_depth,
            )

        return NNVVCDecodeResult(
            recon_yuv_path=output_recon_path,
            frames_decoded=hdr.frame_count,
            width=w,
            height=h,
            reconstructed_frames=reconstructed_frames,
        )

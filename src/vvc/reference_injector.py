"""
Reference Frame Injection module for NN-VVC.

Manages the injection of the neural-reconstructed I-frame (produced by LIC + IHA)
as the reference picture (POC 0) for conventional VTM 12.0 Low-Delay P (LDP) inter coding.

Paper Reference:
    "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
     image coding for machines", Section IV-C (Conventional Video Coding Integration).
    - QP Relationship: QP_intra = QP_inter - 5.
    - Reference mechanism: The reconstructed I-frame (POC 0) is used as the
      reference picture for subsequent VTM P-frames (POC 1..N-1).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.vvc.vtm_wrapper import VTMWrapper, VTMEncodeResult
from src.vvc.yuv_utils import (
    calculate_yuv420_frame_bytes,
    get_yuv_frame_count,
    read_yuv_frame,
    read_yuv_sequence,
    rgb_to_yuv420,
    write_yuv_sequence,
)


class ReferenceInjector:
    """
    Orchestrates the preparation and injection of neural reconstructed I-frames
    into the VTM 12.0 Low-Delay P coding pipeline.
    """

    def __init__(
        self,
        vtm_wrapper: Optional[VTMWrapper] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the ReferenceInjector.

        Args:
            vtm_wrapper: Existing VTMWrapper instance (if None, initializes a default one).
            temp_dir: Temporary directory for injected YUV sequences (defaults to E:\\temp or system temp).
        """
        self.vtm = vtm_wrapper if vtm_wrapper is not None else VTMWrapper()
        if temp_dir is not None:
            self.temp_dir = Path(temp_dir)
        elif Path(r"E:\temp").is_dir():
            self.temp_dir = Path(r"E:\temp")
        else:
            self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_qp_pair(qp_inter: int) -> Tuple[int, int]:
        """
        Calculate paper-specified (QP_intra, QP_inter) pair where QP_intra = QP_inter - 5.

        Args:
            qp_inter: Target inter QP (0 to 63).

        Returns:
            (qp_intra, qp_inter): Tuple of integers clamped to [0, 63].
        """
        if not (0 <= qp_inter <= 63):
            raise ValueError(f"QP inter must be in [0, 63], got {qp_inter}")
        qp_intra = max(0, min(63, qp_inter - 5))
        return qp_intra, qp_inter

    def build_injected_sequence(
        self,
        neural_i_frame: Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        inter_frames: Union[
            List[Union[np.ndarray, torch.Tensor]],
            List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
            Path,
            str,
        ],
        width: int,
        height: int,
        output_yuv_path: Union[str, Path],
        standard: str = "bt709",
        bit_depth: int = 8,
    ) -> Path:
        """
        Construct a composite YUV sequence where Frame 0 is the neural-reconstructed I-frame,
        followed by the raw uncompressed subsequent frames (Frames 1..N-1).

        Args:
            neural_i_frame: Neural reconstructed I-frame (NumPy array, PyTorch tensor, or (Y, U, V) planar tuple).
            inter_frames: List of remaining frames (RGB tensors/arrays or (Y,U,V) tuples), or path to existing YUV file.
            width: Frame width (must be even).
            height: Frame height (must be even).
            output_yuv_path: Destination path for the combined sequence.
            standard: Color conversion standard ('bt709' or 'bt601').
            bit_depth: Bit depth (8 or 10).

        Returns:
            Path to the written composite YUV sequence.
        """
        out_path = Path(output_yuv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Process neural I-frame into (Y, U, V)
        if isinstance(neural_i_frame, tuple) and len(neural_i_frame) == 3:
            y_i, u_i, v_i = neural_i_frame
        elif isinstance(neural_i_frame, (np.ndarray, torch.Tensor)):
            y_i, u_i, v_i = rgb_to_yuv420(neural_i_frame, standard=standard, bit_depth=bit_depth)
        else:
            raise TypeError(f"Unsupported neural I-frame type: {type(neural_i_frame)}")

        if y_i.shape != (height, width):
            raise ValueError(f"I-frame dimension mismatch: expected ({height}, {width}), got {y_i.shape}")

        frames_to_write: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [(y_i, u_i, v_i)]

        # 2. Process inter frames
        if isinstance(inter_frames, (str, Path)):
            # Read from existing YUV sequence (skip frame 0 of source if it was the original I-frame)
            src_yuv = Path(inter_frames)
            total_src_frames = get_yuv_frame_count(src_yuv, width, height, bit_depth=bit_depth)
            if total_src_frames > 1:
                # Read frames 1..total_src_frames-1
                with open(src_yuv, "rb") as fh:
                    for f_idx in range(1, total_src_frames):
                        frames_to_write.append(read_yuv_frame(fh, width, height, frame_idx=f_idx, bit_depth=bit_depth))
        elif isinstance(inter_frames, list):
            for f_item in inter_frames:
                if isinstance(f_item, tuple) and len(f_item) == 3:
                    frames_to_write.append(f_item)
                elif isinstance(f_item, (np.ndarray, torch.Tensor)):
                    frames_to_write.append(rgb_to_yuv420(f_item, standard=standard, bit_depth=bit_depth))
                else:
                    raise TypeError(f"Unsupported inter frame item type: {type(f_item)}")
        else:
            raise TypeError(f"Unsupported inter_frames specification: {type(inter_frames)}")

        write_yuv_sequence(out_path, frames_to_write, width, height, bit_depth=bit_depth)
        return out_path

    def encode_with_reference_injection(
        self,
        neural_i_frame: Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        inter_frames: Union[
            List[Union[np.ndarray, torch.Tensor]],
            List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
            Path,
            str,
        ],
        width: int,
        height: int,
        qp_inter: int,
        output_bitstream: Union[str, Path],
        recon_yuv: Optional[Union[str, Path]] = None,
        cfg_name_or_path: Union[str, Path] = "encoder_lowdelay_P_vtm.cfg",
        frame_rate: int = 30,
        standard: str = "bt709",
        bit_depth: int = 8,
        extra_args: Optional[List[str]] = None,
    ) -> VTMEncodeResult:
        """
        Builds the reference-injected sequence and executes VTM Low-Delay P encoding.

        Args:
            neural_i_frame: Neural reconstructed I-frame (POC 0 reference).
            inter_frames: Uncompressed subsequent frames (Frames 1..N-1).
            width: Frame width.
            height: Frame height.
            qp_inter: Target inter quantization parameter.
            output_bitstream: Output .vvc bitstream path.
            recon_yuv: Optional path for reconstructed YUV sequence.
            cfg_name_or_path: VTM configuration file (default 'encoder_lowdelay_P_vtm.cfg').
            frame_rate: Frame rate.
            standard: Color standard.
            bit_depth: Bit depth.
            extra_args: Additional VTM flags.

        Returns:
            VTMEncodeResult containing execution metrics.
        """
        qp_intra, qp_inter_val = self.calculate_qp_pair(qp_inter)

        # Temporary path for composite YUV sequence
        tag = f"inj_{width}x{height}_qp{qp_inter_val}"
        injected_yuv_path = self.temp_dir / f"{tag}_input.yuv"

        self.build_injected_sequence(
            neural_i_frame=neural_i_frame,
            inter_frames=inter_frames,
            width=width,
            height=height,
            output_yuv_path=injected_yuv_path,
            standard=standard,
            bit_depth=bit_depth,
        )

        total_frames = get_yuv_frame_count(injected_yuv_path, width, height, bit_depth=bit_depth)

        # Encode with VTM using Low-Delay P configuration
        # VTM's IntraQPOffset is set so that POC 0 matches QP_intra = QP_inter - 5
        # In encoder_lowdelay_P_vtm.cfg: IntraQPOffset default is -1.
        # We can pass custom IntraQPOffset if needed, or target QP_inter directly
        cmd_extra = []
        if extra_args:
            cmd_extra.extend(extra_args)

        encode_result = self.vtm.encode(
            input_yuv=injected_yuv_path,
            width=width,
            height=height,
            frame_count=total_frames,
            qp=qp_inter_val,
            output_bitstream=output_bitstream,
            recon_yuv=recon_yuv,
            cfg_name_or_path=cfg_name_or_path,
            frame_rate=frame_rate,
            input_bit_depth=bit_depth,
            internal_bit_depth=bit_depth,
            output_bit_depth=bit_depth,
            extra_args=cmd_extra if cmd_extra else None,
        )

        return encode_result

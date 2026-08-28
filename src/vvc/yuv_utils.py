"""
YUV and spatial utility module for NN-VVC.

Provides conversion between RGB (PyTorch tensors / NumPy arrays) and planar
YUV420 representations, color matrix transformations (BT.601 and BT.709),
raw planar file I/O compatible with VTM 12.0, and spatial padding/unpadding.

Paper Reference:
    "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
     image coding for machines", Section IV (Conventional Video Coding Integration).
"""

from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Color Matrix Definitions (ITU-R BT.601 and ITU-R BT.709)
# Full-range coefficients normalized for [0, 1] floating point RGB/YUV
# -----------------------------------------------------------------------------

# BT.601 conversion constants
# Y = 0.299*R + 0.587*G + 0.114*B
# U = (B - Y) / 1.772 + 0.5 = -0.168736*R - 0.331264*G + 0.5*B + 0.5
# V = (R - Y) / 1.402 + 0.5 =  0.5*R - 0.418688*G - 0.081312*B + 0.5
_BT601_RGB2YUV = np.array(
    [
        [0.299000, 0.587000, 0.114000],
        [-0.168736, -0.331264, 0.500000],
        [0.500000, -0.418688, -0.081312],
    ],
    dtype=np.float64,
)

_BT601_YUV2RGB = np.array(
    [
        [1.0, 0.0, 1.402000],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772000, 0.0],
    ],
    dtype=np.float64,
)

# BT.709 conversion constants (standard for HD / JVET test material)
# Y = 0.2126*R + 0.7152*G + 0.0722*B
# U = (B - Y) / 1.8556 + 0.5 = -0.114572*R - 0.385428*G + 0.5*B + 0.5
# V = (R - Y) / 1.5748 + 0.5 =  0.5*R - 0.454153*G - 0.045847*B + 0.5
_BT709_RGB2YUV = np.array(
    [
        [0.212600, 0.715200, 0.072200],
        [-0.114572, -0.385428, 0.500000],
        [0.500000, -0.454153, -0.045847],
    ],
    dtype=np.float64,
)

_BT709_YUV2RGB = np.array(
    [
        [1.0, 0.0, 1.574800],
        [1.0, -0.187324, -0.468124],
        [1.0, 1.855600, 0.0],
    ],
    dtype=np.float64,
)


def _get_matrices(standard: str) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve forward and inverse color matrices for the specified standard."""
    std = standard.lower().replace("-", "").replace("_", "")
    if std in ("bt601", "601"):
        return _BT601_RGB2YUV, _BT601_YUV2RGB
    elif std in ("bt709", "709", "rec709"):
        return _BT709_RGB2YUV, _BT709_YUV2RGB
    else:
        raise ValueError(
            f"Unsupported color standard '{standard}'. Choose 'bt601' or 'bt709'."
        )


# -----------------------------------------------------------------------------
# Spatial Padding & Dimension Alignment
# -----------------------------------------------------------------------------

def pad_to_alignment(
    img: Union[np.ndarray, torch.Tensor],
    align: int = 2,
    mode: str = "reflect",
) -> Tuple[Union[np.ndarray, torch.Tensor], Tuple[int, int]]:
    """
    Pad an image or tensor so that height and width are divisible by `align`.

    Args:
        img: Input image as NumPy array (H, W, C) or (H, W), or PyTorch tensor (C, H, W) or (B, C, H, W).
        align: Required spatial alignment factor (default 2 for YUV420; e.g. 16 or 32 for neural codecs).
        mode: Padding mode ('reflect', 'replicate', or 'constant').

    Returns:
        padded_img: Padded image/tensor with dimensions divisible by `align`.
        original_shape: Original (height, width) tuple for unpadding.
    """
    if align <= 1:
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2], img.shape[-1]
        else:
            h, w = img.shape[0], img.shape[1]
        return img, (h, w)

    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2], img.shape[-1]
        pad_h = (align - (h % align)) % align
        pad_w = (align - (w % align)) % align
        if pad_h == 0 and pad_w == 0:
            return img, (h, w)
        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        # We pad right and bottom to preserve top-left alignment
        pad_tuple = (0, pad_w, 0, pad_h)
        # If tensor has 3 dims (C, H, W), unsqueeze to 4D for reflection pad if needed
        is_3d = img.ndim == 3
        t = img.unsqueeze(0) if is_3d else img
        padded_t = F.pad(t, pad_tuple, mode=mode)
        padded = padded_t.squeeze(0) if is_3d else padded_t
        return padded, (h, w)
    else:
        h, w = img.shape[0], img.shape[1]
        pad_h = (align - (h % align)) % align
        pad_w = (align - (w % align)) % align
        if pad_h == 0 and pad_w == 0:
            return img, (h, w)
        if img.ndim == 3:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        elif img.ndim == 2:
            pad_width = ((0, pad_h), (0, pad_w))
        else:
            raise ValueError(f"Unsupported NumPy image shape: {img.shape}")
        
        np_mode = "reflect" if mode == "reflect" else ("edge" if mode == "replicate" else "constant")
        padded = np.pad(img, pad_width, mode=np_mode)
        return padded, (h, w)


def unpad_from_alignment(
    padded_img: Union[np.ndarray, torch.Tensor],
    original_shape: Tuple[int, int],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Restore original spatial dimensions from a previously padded image/tensor.

    Args:
        padded_img: Padded NumPy array (H_pad, W_pad, ...) or PyTorch tensor (..., H_pad, W_pad).
        original_shape: Tuple of (original_height, original_width).

    Returns:
        Cropped image/tensor matching original_shape.
    """
    orig_h, orig_w = original_shape
    if isinstance(padded_img, torch.Tensor):
        return padded_img[..., :orig_h, :orig_w]
    else:
        return padded_img[:orig_h, :orig_w, ...]


# -----------------------------------------------------------------------------
# Frame Byte Sizing
# -----------------------------------------------------------------------------

def calculate_yuv420_frame_bytes(
    width: int,
    height: int,
    bit_depth: int = 8,
) -> int:
    """
    Calculate the byte size of a single raw planar YUV420 frame.

    Args:
        width: Frame width (must be positive even integer).
        height: Frame height (must be positive even integer).
        bit_depth: Bit depth (8 or 10-16).

    Returns:
        Total bytes per frame.
    """
    if width <= 0 or width % 2 != 0:
        raise ValueError(f"Width must be a positive even integer, got {width}")
    if height <= 0 or height % 2 != 0:
        raise ValueError(f"Height must be a positive even integer, got {height}")

    bytes_per_sample = 1 if bit_depth == 8 else 2
    # Y plane: W * H
    # U plane: (W // 2) * (H // 2)
    # V plane: (W // 2) * (H // 2)
    total_samples = width * height + 2 * ((width // 2) * (height // 2))
    return total_samples * bytes_per_sample


def get_yuv_frame_count(
    path: Union[str, Path],
    width: int,
    height: int,
    bit_depth: int = 8,
) -> int:
    """
    Calculate number of complete YUV420 frames in a raw planar file.

    Args:
        path: Path to the .yuv file.
        width: Frame width.
        height: Frame height.
        bit_depth: Bit depth (8 or 10-16).

    Returns:
        Integer frame count.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"YUV file not found: {file_path}")

    frame_bytes = calculate_yuv420_frame_bytes(width, height, bit_depth=bit_depth)
    file_bytes = file_path.stat().st_size
    return file_bytes // frame_bytes


# -----------------------------------------------------------------------------
# RGB <-> YUV420 Planar Conversion
# -----------------------------------------------------------------------------

def rgb_to_yuv420(
    rgb: Union[np.ndarray, torch.Tensor],
    standard: str = "bt709",
    bit_depth: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image (NumPy array or PyTorch tensor) into planar YUV420 arrays.

    Args:
        rgb: RGB image.
             - If NumPy array: shape (H, W, 3) in range [0, 255] (uint8) or [0.0, 1.0] (float).
             - If PyTorch tensor: shape (3, H, W) or (1, 3, H, W) in range [0.0, 1.0].
        standard: Color standard ('bt709' or 'bt601'). Default is 'bt709'.
        bit_depth: Output YUV bit depth (8 or 10). Default is 8.

    Returns:
        (Y, U, V): Tuple of planar NumPy arrays:
            - Y: shape (H, W), dtype uint8 (8-bit) or uint16 (10-bit)
            - U: shape (H // 2, W // 2), dtype uint8 or uint16
            - V: shape (H // 2, W // 2), dtype uint8 or uint16
    """
    # Convert PyTorch tensor to NumPy (H, W, 3) float64 in [0, 1]
    if isinstance(rgb, torch.Tensor):
        t = rgb.detach().cpu()
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError(f"Batch dimension > 1 not supported in single-frame rgb_to_yuv420, got shape {t.shape}")
            t = t.squeeze(0)
        if t.ndim != 3 or t.shape[0] != 3:
            raise ValueError(f"Expected tensor shape (3, H, W) or (1, 3, H, W), got {rgb.shape}")
        arr = t.permute(1, 2, 0).numpy().astype(np.float64)
    elif isinstance(rgb, np.ndarray):
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected NumPy array with shape (H, W, 3), got {rgb.shape}")
        if rgb.dtype == np.uint8:
            arr = rgb.astype(np.float64) / 255.0
        else:
            arr = rgb.astype(np.float64)
    else:
        raise TypeError(f"Unsupported RGB input type: {type(rgb)}")

    h, w, _ = arr.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Image dimensions must be even for YUV420 conversion, got {w}x{h}. Use pad_to_alignment first.")

    m_fwd, _ = _get_matrices(standard)

    # Matrix multiplication: [H, W, 3] x [3, 3].T -> [H, W, 3]
    # arr: R, G, B in [0, 1]
    y = arr[:, :, 0] * m_fwd[0, 0] + arr[:, :, 1] * m_fwd[0, 1] + arr[:, :, 2] * m_fwd[0, 2]
    u = arr[:, :, 0] * m_fwd[1, 0] + arr[:, :, 1] * m_fwd[1, 1] + arr[:, :, 2] * m_fwd[1, 2] + 0.5
    v = arr[:, :, 0] * m_fwd[2, 0] + arr[:, :, 1] * m_fwd[2, 1] + arr[:, :, 2] * m_fwd[2, 2] + 0.5

    # 4:2:0 Chroma subsampling via 2x2 box filter (averaging)
    u_sub = (u[0::2, 0::2] + u[0::2, 1::2] + u[1::2, 0::2] + u[1::2, 1::2]) * 0.25
    v_sub = (v[0::2, 0::2] + v[0::2, 1::2] + v[1::2, 0::2] + v[1::2, 1::2]) * 0.25

    # Quantize according to bit depth
    max_val = (1 << bit_depth) - 1
    if bit_depth == 8:
        y_out = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
        u_out = np.clip(np.round(u_sub * 255.0), 0, 255).astype(np.uint8)
        v_out = np.clip(np.round(v_sub * 255.0), 0, 255).astype(np.uint8)
    else:
        scale = float(max_val)
        y_out = np.clip(np.round(y * scale), 0, max_val).astype(np.uint16)
        u_out = np.clip(np.round(u_sub * scale), 0, max_val).astype(np.uint16)
        v_out = np.clip(np.round(v_sub * scale), 0, max_val).astype(np.uint16)

    return y_out, u_out, v_out


def yuv420_to_rgb(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    standard: str = "bt709",
    bit_depth: int = 8,
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert planar YUV420 arrays into an RGB image.

    Args:
        y: Y plane, shape (H, W), dtype uint8 or uint16.
        u: U plane, shape (H // 2, W // 2), dtype uint8 or uint16.
        v: V plane, shape (H // 2, W // 2), dtype uint8 or uint16.
        standard: Color standard ('bt709' or 'bt601'). Default is 'bt709'.
        bit_depth: Bit depth of input YUV (8 or 10). Default is 8.
        return_tensor: If True, returns PyTorch float tensor (3, H, W) in [0.0, 1.0].
                       If False, returns NumPy float32 array (H, W, 3) in [0.0, 1.0].

    Returns:
        RGB image as NumPy float32 array or PyTorch float tensor.
    """
    h, w = y.shape
    if u.shape != (h // 2, w // 2) or v.shape != (h // 2, w // 2):
        raise ValueError(
            f"Chroma plane shape mismatch: Y is {y.shape}, U is {u.shape}, V is {v.shape}."
        )

    max_val = float((1 << bit_depth) - 1)
    y_norm = y.astype(np.float64) / max_val
    u_norm = u.astype(np.float64) / max_val - 0.5
    v_norm = v.astype(np.float64) / max_val - 0.5

    # Chroma upsampling from (H//2, W//2) to (H, W) using bilinear interpolation
    # Repeat elements 2x in each dimension for standard nearest/box, or linear interpolation
    u_up = np.repeat(np.repeat(u_norm, 2, axis=0), 2, axis=1)
    v_up = np.repeat(np.repeat(v_norm, 2, axis=0), 2, axis=1)

    _, m_inv = _get_matrices(standard)

    r = y_norm * m_inv[0, 0] + u_up * m_inv[0, 1] + v_up * m_inv[0, 2]
    g = y_norm * m_inv[1, 0] + u_up * m_inv[1, 1] + v_up * m_inv[1, 2]
    b = y_norm * m_inv[2, 0] + u_up * m_inv[2, 1] + v_up * m_inv[2, 2]

    rgb_arr = np.stack([np.clip(r, 0.0, 1.0), np.clip(g, 0.0, 1.0), np.clip(b, 0.0, 1.0)], axis=-1).astype(np.float32)

    if return_tensor:
        # Permute (H, W, 3) -> (3, H, W)
        return torch.from_numpy(rgb_arr).permute(2, 0, 1)
    return rgb_arr


# -----------------------------------------------------------------------------
# Planar YUV Binary File I/O
# -----------------------------------------------------------------------------

def read_yuv_frame(
    file_or_path: Union[str, Path, BinaryIO],
    width: int,
    height: int,
    frame_idx: int = 0,
    bit_depth: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a single planar YUV420 frame from an open file or file path.

    Args:
        file_or_path: File path (str/Path) or open binary file handle.
        width: Frame width.
        height: Frame height.
        frame_idx: 0-indexed frame index to read.
        bit_depth: Bit depth (8 or 10).

    Returns:
        (Y, U, V): Planar NumPy arrays.
    """
    frame_bytes = calculate_yuv420_frame_bytes(width, height, bit_depth=bit_depth)
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    samples_y = y_size
    samples_uv = uv_size

    def _read_from_handle(fh: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fh.seek(frame_idx * frame_bytes)
        raw = fh.read(frame_bytes)
        if len(raw) < frame_bytes:
            raise EOFError(
                f"Requested frame {frame_idx} (offset {frame_idx * frame_bytes}) but file has only {len(raw)} bytes remaining."
            )
        data = np.frombuffer(raw, dtype=dtype)
        y = data[:samples_y].reshape((height, width))
        u = data[samples_y : samples_y + samples_uv].reshape((height // 2, width // 2))
        v = data[samples_y + samples_uv :].reshape((height // 2, width // 2))
        return y, u, v

    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as fh:
            return _read_from_handle(fh)
    else:
        return _read_from_handle(file_or_path)


def write_yuv_frame(
    file_or_path: Union[str, Path, BinaryIO],
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bit_depth: int = 8,
    append: bool = False,
) -> None:
    """
    Write a single planar YUV420 frame to a file or file handle.

    Args:
        file_or_path: Destination path or open binary file handle.
        y: Y plane.
        u: U plane.
        v: V plane.
        bit_depth: Bit depth (8 or 10).
        append: If True and writing to a file path, appends to file.
    """
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    raw_bytes = y.astype(dtype).tobytes() + u.astype(dtype).tobytes() + v.astype(dtype).tobytes()

    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if append else "wb"
        with open(p, mode) as fh:
            fh.write(raw_bytes)
    else:
        file_or_path.write(raw_bytes)


def read_yuv_sequence(
    path: Union[str, Path],
    width: int,
    height: int,
    frame_count: Optional[int] = None,
    bit_depth: int = 8,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read multiple sequential planar YUV420 frames from a .yuv file.

    Args:
        path: Path to the .yuv file.
        width: Frame width.
        height: Frame height.
        frame_count: Number of frames to read (if None, reads all available frames).
        bit_depth: Bit depth (8 or 10).

    Returns:
        List of (Y, U, V) tuples.
    """
    total_available = get_yuv_frame_count(path, width, height, bit_depth=bit_depth)
    n_frames = min(total_available, frame_count) if frame_count is not None else total_available

    frames = []
    with open(path, "rb") as fh:
        for idx in range(n_frames):
            frames.append(read_yuv_frame(fh, width, height, frame_idx=idx, bit_depth=bit_depth))
    return frames


def write_yuv_sequence(
    path: Union[str, Path],
    frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    width: int,
    height: int,
    bit_depth: int = 8,
) -> None:
    """
    Write a list of planar YUV420 frames to a raw .yuv sequence file.

    Args:
        path: Destination path.
        frames: List of (Y, U, V) tuples.
        width: Frame width.
        height: Frame height.
        bit_depth: Bit depth (8 or 10).
    """
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, "wb") as fh:
        for y, u, v in frames:
            if y.shape != (height, width):
                raise ValueError(f"Frame dimension mismatch: expected ({height}, {width}), got {y.shape}")
            write_yuv_frame(fh, y, u, v, bit_depth=bit_depth)


# -----------------------------------------------------------------------------
# PyTorch Tensor Batch Helpers
# -----------------------------------------------------------------------------

def tensor_to_yuv420_bytes(
    tensor: torch.Tensor,
    standard: str = "bt709",
    bit_depth: int = 8,
) -> bytes:
    """
    Convert a PyTorch RGB tensor (3, H, W) or (1, 3, H, W) in [0.0, 1.0] to raw planar YUV420 bytes.
    """
    y, u, v = rgb_to_yuv420(tensor, standard=standard, bit_depth=bit_depth)
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    return y.astype(dtype).tobytes() + u.astype(dtype).tobytes() + v.astype(dtype).tobytes()


def yuv420_bytes_to_tensor(
    raw_bytes: bytes,
    width: int,
    height: int,
    standard: str = "bt709",
    bit_depth: int = 8,
) -> torch.Tensor:
    """
    Convert raw planar YUV420 bytes of a single frame into a PyTorch RGB tensor (3, H, W) in [0.0, 1.0].
    """
    expected_len = calculate_yuv420_frame_bytes(width, height, bit_depth=bit_depth)
    if len(raw_bytes) < expected_len:
        raise ValueError(f"Byte buffer length {len(raw_bytes)} is shorter than expected frame size {expected_len}.")

    dtype = np.uint8 if bit_depth == 8 else np.uint16
    data = np.frombuffer(raw_bytes[:expected_len], dtype=dtype)
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    y = data[:y_size].reshape((height, width))
    u = data[y_size : y_size + uv_size].reshape((height // 2, width // 2))
    v = data[y_size + uv_size :].reshape((height // 2, width // 2))

    return yuv420_to_rgb(y, u, v, standard=standard, bit_depth=bit_depth, return_tensor=True)

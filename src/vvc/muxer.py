"""
Hybrid NN-VVC Bitstream Muxer and Demuxer.

Provides serialization and deserialization for the combined .nnvvc hybrid container
carrying both the neural I-frame representation (LIC/IHA) and the conventional
VTM 12.0 P-frame .vvc Annex-B bitstream.

Paper Reference:
    "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
     image coding for machines", Section IV (Conventional Video Coding Integration).

Engineering Note:
    The .nnvvc byte-level binary container format is a project-specific engineering
    format created to represent the hybrid bitstream concept described by the NN-VVC
    research paper. It is deterministic, versioned, checksummed, and fully reversible.
"""

import io
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union


class NNVVCFormatError(Exception):
    """Base exception for NN-VVC container errors."""
    pass


class NNVVCInvalidMagicError(NNVVCFormatError):
    """Raised when the container does not begin with the valid NNVVC magic bytes."""
    pass


class NNVVCUnsupportedVersionError(NNVVCFormatError):
    """Raised when the container version is not supported."""
    pass


class NNVVCTruncatedFileError(NNVVCFormatError):
    """Raised when the container file is truncated or payload lengths exceed file size."""
    pass


class NNVVCCorruptedDataError(NNVVCFormatError):
    """Raised when payload checksum verification fails."""
    pass


@dataclass
class NNVVCContainerHeader:
    """
    Structured metadata header for .nnvvc hybrid bitstream container.
    """
    version: int = 1
    width: int = 0
    height: int = 0
    frame_count: int = 0
    framerate: int = 30
    bit_depth: int = 8
    chroma_format: int = 420
    qp_intra: int = 27
    qp_inter: int = 32
    scale_num: int = 1
    scale_denom: int = 1
    neural_payload_size: int = 0
    vtm_payload_size: int = 0
    payload_crc32: int = 0
    extra_metadata: bytes = b""

    # Constants
    MAGIC: bytes = b"NNVVC"
    CURRENT_VERSION: int = 1
    # Binary layout:
    # 5s: Magic (b"NNVVC")
    # H : Version (uint16)
    # I : Header Size (uint32)
    # I : Width (uint32)
    # I : Height (uint32)
    # I : Frame Count (uint32)
    # H : Framerate (uint16)
    # B : Bit Depth (uint8)
    # H : Chroma Format (uint16)
    # B : QP Intra (uint8)
    # B : QP Inter (uint8)
    # B : Scale Num (uint8)
    # B : Scale Denom (uint8)
    # Q : Neural Payload Size (uint64)
    # Q : VTM Payload Size (uint64)
    # I : Payload CRC32 (uint32)
    # I : Extra Metadata Size (uint32)
    # Total fixed fields size = 5 + 2 + 4 + 4 + 4 + 4 + 2 + 1 + 2 + 1 + 1 + 1 + 1 + 8 + 8 + 4 + 4 = 56 bytes
    HEADER_STRUCT_FORMAT: str = "!5sHIIIIHBHBBBBQQII"
    FIXED_HEADER_SIZE: int = struct.calcsize(HEADER_STRUCT_FORMAT)

    def validate(self) -> None:
        """Validate header fields."""
        if self.version != self.CURRENT_VERSION:
            raise NNVVCUnsupportedVersionError(
                f"Unsupported container version {self.version}. Supported versions: [{self.CURRENT_VERSION}]."
            )
        if self.width <= 0 or self.width % 2 != 0:
            raise ValueError(f"Width must be a positive even integer, got {self.width}")
        if self.height <= 0 or self.height % 2 != 0:
            raise ValueError(f"Height must be a positive even integer, got {self.height}")
        if self.frame_count <= 0:
            raise ValueError(f"Frame count must be positive, got {self.frame_count}")
        if self.framerate <= 0:
            raise ValueError(f"Framerate must be positive, got {self.framerate}")
        if self.bit_depth not in (8, 10, 12, 16):
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")
        if self.chroma_format not in (400, 420, 422, 444):
            raise ValueError(f"Unsupported chroma format: {self.chroma_format}")
        if not (0 <= self.qp_intra <= 63):
            raise ValueError(f"QP intra out of range: {self.qp_intra}")
        if not (0 <= self.qp_inter <= 63):
            raise ValueError(f"QP inter out of range: {self.qp_inter}")
        if self.scale_num <= 0 or self.scale_denom <= 0:
            raise ValueError(f"Scale factors must be positive: {self.scale_num}/{self.scale_denom}")
        if self.neural_payload_size < 0 or self.vtm_payload_size < 0:
            raise ValueError("Payload sizes cannot be negative")

    def serialize(self) -> bytes:
        """Serialize header into binary bytes."""
        self.validate()
        extra_len = len(self.extra_metadata)
        header_total_size = self.FIXED_HEADER_SIZE + extra_len

        fixed_bytes = struct.pack(
            self.HEADER_STRUCT_FORMAT,
            self.MAGIC,
            self.version,
            header_total_size,
            self.width,
            self.height,
            self.frame_count,
            self.framerate,
            self.bit_depth,
            self.chroma_format,
            self.qp_intra,
            self.qp_inter,
            self.scale_num,
            self.scale_denom,
            self.neural_payload_size,
            self.vtm_payload_size,
            self.payload_crc32,
            extra_len,
        )
        return fixed_bytes + self.extra_metadata

    @classmethod
    def deserialize(cls, data: bytes) -> Tuple["NNVVCContainerHeader", int]:
        """
        Deserialize header from binary bytes.

        Returns:
            (header_instance, total_header_bytes_consumed)
        """
        if len(data) < cls.FIXED_HEADER_SIZE:
            raise NNVVCTruncatedFileError(
                f"Data length ({len(data)} bytes) is smaller than fixed header size ({cls.FIXED_HEADER_SIZE} bytes)."
            )

        (
            magic,
            version,
            header_total_size,
            width,
            height,
            frame_count,
            framerate,
            bit_depth,
            chroma_format,
            qp_intra,
            qp_inter,
            scale_num,
            scale_denom,
            neural_size,
            vtm_size,
            crc32,
            extra_len,
        ) = struct.unpack_from(cls.HEADER_STRUCT_FORMAT, data, 0)

        if magic != cls.MAGIC:
            raise NNVVCInvalidMagicError(f"Invalid magic identifier {magic!r}, expected {cls.MAGIC!r}.")
        if version != cls.CURRENT_VERSION:
            raise NNVVCUnsupportedVersionError(
                f"Unsupported container version {version}. Supported versions: [{cls.CURRENT_VERSION}]."
            )
        if len(data) < header_total_size:
            raise NNVVCTruncatedFileError(
                f"Data length ({len(data)} bytes) is smaller than declared total header size ({header_total_size} bytes)."
            )

        extra_metadata = data[cls.FIXED_HEADER_SIZE : cls.FIXED_HEADER_SIZE + extra_len]

        header = cls(
            version=version,
            width=width,
            height=height,
            frame_count=frame_count,
            framerate=framerate,
            bit_depth=bit_depth,
            chroma_format=chroma_format,
            qp_intra=qp_intra,
            qp_inter=qp_inter,
            scale_num=scale_num,
            scale_denom=scale_denom,
            neural_payload_size=neural_size,
            vtm_payload_size=vtm_size,
            payload_crc32=crc32,
            extra_metadata=extra_metadata,
        )
        header.validate()
        return header, header_total_size


@dataclass
class NNVVCPayload:
    """
    Decoded contents of an .nnvvc hybrid container.
    """
    header: NNVVCContainerHeader
    neural_payload: bytes
    vtm_payload: bytes

    @property
    def total_payload_bytes(self) -> int:
        return len(self.neural_payload) + len(self.vtm_payload)


class NNVVCMuxer:
    """
    Multiplexer for packaging neural I-frame payload, VTM P-frame payload,
    and sequence metadata into a .nnvvc container.
    """

    @staticmethod
    def calculate_crc32(neural_payload: bytes, vtm_payload: bytes) -> int:
        """Compute CRC32 checksum over combined payloads for tamper/truncation detection."""
        crc = zlib.crc32(neural_payload)
        return zlib.crc32(vtm_payload, crc)

    @classmethod
    def mux(
        cls,
        header: NNVVCContainerHeader,
        neural_payload: bytes,
        vtm_payload: bytes,
        output_path_or_handle: Optional[Union[str, Path, BinaryIO]] = None,
    ) -> bytes:
        """
        Package metadata, neural I-frame payload, and VTM bitstream into .nnvvc binary format.

        Args:
            header: Container header with sequence metadata.
            neural_payload: Raw bytes of neural I-frame representation.
            vtm_payload: Raw bytes of VTM .vvc bitstream.
            output_path_or_handle: Optional destination file path or open binary file handle.

        Returns:
            Complete binary bytes of the multiplexed container.
        """
        # Update payload sizes and checksum
        header.neural_payload_size = len(neural_payload)
        header.vtm_payload_size = len(vtm_payload)
        header.payload_crc32 = cls.calculate_crc32(neural_payload, vtm_payload)

        header_bytes = header.serialize()
        full_container = header_bytes + neural_payload + vtm_payload

        if output_path_or_handle is not None:
            if isinstance(output_path_or_handle, (str, Path)):
                out_p = Path(output_path_or_handle)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "wb") as f:
                    f.write(full_container)
            else:
                output_path_or_handle.write(full_container)

        return full_container


class NNVVCDeMuxer:
    """
    Demultiplexer for parsing and extracting metadata and bitstreams from .nnvvc containers.
    """

    @classmethod
    def read_header(
        cls,
        input_path_or_bytes: Union[str, Path, bytes, BinaryIO],
    ) -> NNVVCContainerHeader:
        """
        Extract only the header metadata without loading full payloads into memory.
        """
        if isinstance(input_path_or_bytes, (str, Path)):
            with open(input_path_or_bytes, "rb") as f:
                fixed_data = f.read(NNVVCContainerHeader.FIXED_HEADER_SIZE)
                if len(fixed_data) < NNVVCContainerHeader.FIXED_HEADER_SIZE:
                    raise NNVVCTruncatedFileError("File too small to contain valid header.")
                header_total_size = struct.unpack_from("!I", fixed_data, 7)[0]
                extra_needed = header_total_size - len(fixed_data)
                full_header_bytes = fixed_data + (f.read(extra_needed) if extra_needed > 0 else b"")
                header, _ = NNVVCContainerHeader.deserialize(full_header_bytes)
                return header
        elif isinstance(input_path_or_bytes, bytes):
            header, _ = NNVVCContainerHeader.deserialize(input_path_or_bytes)
            return header
        else:
            # File handle
            pos = input_path_or_bytes.tell()
            fixed_data = input_path_or_bytes.read(NNVVCContainerHeader.FIXED_HEADER_SIZE)
            if len(fixed_data) < NNVVCContainerHeader.FIXED_HEADER_SIZE:
                raise NNVVCTruncatedFileError("File handle too small to contain valid header.")
            header_total_size = struct.unpack_from("!I", fixed_data, 7)[0]
            extra_needed = header_total_size - len(fixed_data)
            full_header_bytes = fixed_data + (input_path_or_bytes.read(extra_needed) if extra_needed > 0 else b"")
            input_path_or_bytes.seek(pos)
            header, _ = NNVVCContainerHeader.deserialize(full_header_bytes)
            return header

    @classmethod
    def demux(
        cls,
        input_path_or_bytes: Union[str, Path, bytes, BinaryIO],
        verify_checksum: bool = True,
    ) -> NNVVCPayload:
        """
        Demultiplex an .nnvvc container into its constituent header, neural payload, and VTM payload.

        Args:
            input_path_or_bytes: Path to .nnvvc file, raw bytes, or open binary file handle.
            verify_checksum: If True, validates payload CRC32 checksum against header.

        Returns:
            NNVVCPayload containing header, neural_payload, and vtm_payload.
        """
        if isinstance(input_path_or_bytes, (str, Path)):
            with open(input_path_or_bytes, "rb") as f:
                data = f.read()
        elif isinstance(input_path_or_bytes, bytes):
            data = input_path_or_bytes
        else:
            data = input_path_or_bytes.read()

        header, header_size = NNVVCContainerHeader.deserialize(data)

        expected_total_len = header_size + header.neural_payload_size + header.vtm_payload_size
        if len(data) < expected_total_len:
            raise NNVVCTruncatedFileError(
                f"Container truncated: expected at least {expected_total_len} bytes, got {len(data)} bytes."
            )

        neural_offset = header_size
        neural_end = neural_offset + header.neural_payload_size
        vtm_end = neural_end + header.vtm_payload_size

        neural_payload = data[neural_offset:neural_end]
        vtm_payload = data[neural_end:vtm_end]

        if verify_checksum:
            actual_crc = NNVVCMuxer.calculate_crc32(neural_payload, vtm_payload)
            if actual_crc != header.payload_crc32:
                raise NNVVCCorruptedDataError(
                    f"Checksum mismatch: header CRC32 is {header.payload_crc32}, computed is {actual_crc}."
                )

        return NNVVCPayload(
            header=header,
            neural_payload=neural_payload,
            vtm_payload=vtm_payload,
        )

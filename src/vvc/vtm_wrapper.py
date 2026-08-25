import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class VTMError(Exception):
    """Base exception for VTM-related errors."""
    pass


class VTMNotAvailableError(VTMError):
    """Raised when VTM 12.0 executables or dependencies cannot be found."""
    pass


class VTMVersionError(VTMError):
    """Raised when the detected VTM executable does not match the required version (12.0)."""
    pass


class VTMExecutionError(VTMError):
    """Raised when VTM encoding or decoding process fails."""
    def __init__(
        self,
        message: str,
        executable: Path,
        cmd: List[str],
        returncode: int,
        stdout: str,
        stderr: str,
        expected_output: Optional[Path] = None,
    ):
        super().__init__(message)
        self.executable = executable
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.expected_output = expected_output

    def __str__(self) -> str:
        cmd_str = " ".join(f'"{c}"' if " " in c else c for c in self.cmd)
        return (
            f"{super().__str__()}\n"
            f"Executable: {self.executable}\n"
            f"Command: {cmd_str}\n"
            f"Return code: {self.returncode}\n"
            f"Expected output: {self.expected_output}\n"
            f"--- STDOUT (last 500 chars) ---\n{self.stdout[-500:]}\n"
            f"--- STDERR (last 500 chars) ---\n{self.stderr[-500:]}"
        )


@dataclass
class VTMEncodeResult:
    """Structured result returned by VTMWrapper.encode()."""
    bitstream_path: Path
    recon_path: Optional[Path]
    returncode: int
    stdout: str
    stderr: str
    total_bits: Optional[int] = None
    bitrate_kbps: Optional[float] = None
    psnr_y: Optional[float] = None
    psnr_u: Optional[float] = None
    psnr_v: Optional[float] = None
    psnr_yuv: Optional[float] = None
    encode_time_sec: Optional[float] = None
    frames_encoded: Optional[int] = None


@dataclass
class VTMDecodeResult:
    """Structured result returned by VTMWrapper.decode()."""
    recon_path: Path
    returncode: int
    stdout: str
    stderr: str
    frames_decoded: Optional[int] = None
    decode_time_sec: Optional[float] = None


class VTMWrapper:
    """
    Python wrapper for the official VTM 12.0 video codec (EncoderApp and DecoderApp).

    Paper Reference:
        "NN-VVC: Versatile Video Coding boosted by self-supervisedly learned
         image coding for machines", Section IV (Conventional Video Coding Integration).
    """

    REQUIRED_VERSION_STR = "12.0"
    DEFAULT_VTM_ROOT = Path(r"E:\VTM-12.0")
    DEFAULT_ENCODER = (
        DEFAULT_VTM_ROOT / "bin" / "mgwmake" / "gcc-mingw-13.2" / "x86_64" / "release" / "EncoderApp.exe"
    )
    DEFAULT_DECODER = (
        DEFAULT_VTM_ROOT / "bin" / "mgwmake" / "gcc-mingw-13.2" / "x86_64" / "release" / "DecoderApp.exe"
    )
    DEFAULT_CFG_DIR = DEFAULT_VTM_ROOT / "cfg"
    DEFAULT_MSYS2_BIN = Path(r"C:\msys64\ucrt64\bin")

    def __init__(
        self,
        encoder_path: Optional[Union[str, Path]] = None,
        decoder_path: Optional[Union[str, Path]] = None,
        cfg_dir: Optional[Union[str, Path]] = None,
        auto_validate: bool = True,
    ):
        """
        Initialize the VTM 12.0 wrapper.

        Args:
            encoder_path: Path to EncoderApp.exe (defaults to verified installation).
            decoder_path: Path to DecoderApp.exe (defaults to verified installation).
            cfg_dir: Path to VTM cfg/ directory.
            auto_validate: If True, validate executable presence and versions on init.
        """
        self.encoder_path = Path(encoder_path) if encoder_path else self.DEFAULT_ENCODER
        self.decoder_path = Path(decoder_path) if decoder_path else self.DEFAULT_DECODER
        self.cfg_dir = Path(cfg_dir) if cfg_dir else self.DEFAULT_CFG_DIR

        # Prepare environment with MSYS2 runtime DLLs in PATH if on Windows
        self._env = os.environ.copy()
        if self.DEFAULT_MSYS2_BIN.is_dir():
            msys_str = str(self.DEFAULT_MSYS2_BIN)
            # Ensure MSYS2 UCRT64 bin is at the front of both PATH and Path
            current_path = self._env.get("PATH", "")
            self._env["PATH"] = f"{msys_str};{current_path}"
            self._env["Path"] = f"{msys_str};{self._env.get('Path', '')}"

        if auto_validate:
            self.validate_environment()

    def validate_environment(self) -> None:
        """
        Verify that EncoderApp and DecoderApp exist and are genuine VTM 12.0 executables.

        Raises:
            VTMNotAvailableError: If executables are not found.
            VTMVersionError: If version is not 12.0.
        """
        if not self.encoder_path.is_file():
            raise VTMNotAvailableError(
                f"VTM 12.0 Encoder executable not found at: {self.encoder_path}"
            )
        if not self.decoder_path.is_file():
            raise VTMNotAvailableError(
                f"VTM 12.0 Decoder executable not found at: {self.decoder_path}"
            )

        enc_version = self.get_encoder_version()
        if self.REQUIRED_VERSION_STR not in enc_version:
            raise VTMVersionError(
                f"Encoder at {self.encoder_path} reported version '{enc_version}', "
                f"expected '{self.REQUIRED_VERSION_STR}'."
            )

        dec_version = self.get_decoder_version()
        if self.REQUIRED_VERSION_STR not in dec_version:
            raise VTMVersionError(
                f"Decoder at {self.decoder_path} reported version '{dec_version}', "
                f"expected '{self.REQUIRED_VERSION_STR}'."
            )

    def get_encoder_version(self) -> str:
        """Execute EncoderApp and parse the version banner."""
        if not self.encoder_path.is_file():
            raise VTMNotAvailableError(f"EncoderApp not found at: {self.encoder_path}")

        proc = subprocess.run(
            [str(self.encoder_path), "--help"],
            capture_output=True,
            text=True,
            env=self._env,
        )
        output = proc.stdout + proc.stderr
        for line in output.splitlines():
            if "VVCSoftware: VTM Encoder Version" in line:
                return line.strip()
        return output[:200].strip()

    def get_decoder_version(self) -> str:
        """Execute DecoderApp and parse the version banner."""
        if not self.decoder_path.is_file():
            raise VTMNotAvailableError(f"DecoderApp not found at: {self.decoder_path}")

        proc = subprocess.run(
            [str(self.decoder_path), "--help"],
            capture_output=True,
            text=True,
            env=self._env,
        )
        output = proc.stdout + proc.stderr
        for line in output.splitlines():
            if "VVCSoftware: VTM Decoder Version" in line:
                return line.strip()
        return output[:200].strip()

    def get_config_path(self, cfg_name_or_path: Union[str, Path]) -> Path:
        """Resolve a configuration file name or path."""
        p = Path(cfg_name_or_path)
        if p.is_file():
            return p
        if self.cfg_dir:
            cand = self.cfg_dir / cfg_name_or_path
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"VTM config file not found: {cfg_name_or_path}")

    def encode(
        self,
        input_yuv: Union[str, Path],
        width: int,
        height: int,
        frame_count: int,
        qp: int,
        output_bitstream: Union[str, Path],
        recon_yuv: Optional[Union[str, Path]] = None,
        cfg_name_or_path: Union[str, Path] = "encoder_lowdelay_P_vtm.cfg",
        frame_rate: int = 30,
        input_bit_depth: int = 8,
        internal_bit_depth: int = 8,
        output_bit_depth: int = 8,
        chroma_format: int = 420,
        extra_args: Optional[List[str]] = None,
    ) -> VTMEncodeResult:
        """
        Run VTM 12.0 Encoder on raw YUV sequence.

        Args:
            input_yuv: Path to input YUV file.
            width: Frame width (must be positive even integer).
            height: Frame height (must be positive even integer).
            frame_count: Number of frames to encode (> 0).
            qp: Quantization parameter (0 to 63).
            output_bitstream: Destination path for .vvc bitstream.
            recon_yuv: Optional path to save reconstructed YUV sequence.
            cfg_name_or_path: Configuration file name (e.g. 'encoder_lowdelay_P_vtm.cfg') or full path.
            frame_rate: Frame rate (default 30).
            input_bit_depth: Input bit depth (default 8).
            internal_bit_depth: Internal processing bit depth (default 8).
            output_bit_depth: Output bit depth (default 8).
            chroma_format: Chroma sub-sampling format idc (default 420).
            extra_args: Optional additional command-line flags.

        Returns:
            VTMEncodeResult containing execution details and parsed coding metrics.
        """
        input_yuv = Path(input_yuv)
        output_bitstream = Path(output_bitstream)
        recon_yuv = Path(recon_yuv) if recon_yuv else None

        # Parameter validation
        if not input_yuv.is_file():
            raise FileNotFoundError(f"Input YUV file does not exist: {input_yuv}")
        if width <= 0 or width % 2 != 0:
            raise ValueError(f"Width must be a positive even integer, got {width}")
        if height <= 0 or height % 2 != 0:
            raise ValueError(f"Height must be a positive even integer, got {height}")
        if frame_count <= 0:
            raise ValueError(f"Frame count must be positive, got {frame_count}")
        if not (0 <= qp <= 63):
            raise ValueError(f"QP must be in range [0, 63], got {qp}")

        # Check input file size corresponds to at least frame_count
        frame_bytes = (width * height * 3) // 2 if chroma_format == 420 else width * height * 3
        expected_min_bytes = frame_bytes * frame_count
        actual_bytes = input_yuv.stat().st_size
        if actual_bytes < expected_min_bytes:
            raise ValueError(
                f"Input YUV file size ({actual_bytes} bytes) is smaller than required "
                f"for {frame_count} frames of {width}x{height} YUV{chroma_format} ({expected_min_bytes} bytes)"
            )

        cfg_file = self.get_config_path(cfg_name_or_path)

        # Ensure output directory exists
        output_bitstream.parent.mkdir(parents=True, exist_ok=True)
        if recon_yuv:
            recon_yuv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.encoder_path),
            "-c", str(cfg_file),
            "-i", str(input_yuv),
            "-b", str(output_bitstream),
            "-wdt", str(width),
            "-hgt", str(height),
            "-fr", str(frame_rate),
            "-f", str(frame_count),
            "-q", str(qp),
            f"--InputBitDepth={input_bit_depth}",
            f"--InternalBitDepth={internal_bit_depth}",
            f"--OutputBitDepth={output_bit_depth}",
            f"--InputChromaFormat={chroma_format}",
        ]

        if recon_yuv:
            cmd.extend(["-o", str(recon_yuv)])

        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self._env,
        )

        if proc.returncode != 0:
            raise VTMExecutionError(
                f"VTM Encoder execution failed with code {proc.returncode}",
                executable=self.encoder_path,
                cmd=cmd,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                expected_output=output_bitstream,
            )

        if not output_bitstream.is_file() or output_bitstream.stat().st_size == 0:
            raise VTMExecutionError(
                f"VTM Encoder finished with returncode 0 but bitstream was not created or is empty: {output_bitstream}",
                executable=self.encoder_path,
                cmd=cmd,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                expected_output=output_bitstream,
            )

        # Parse metrics from stdout
        metrics = self.parse_encoder_stdout(proc.stdout)

        return VTMEncodeResult(
            bitstream_path=output_bitstream,
            recon_path=recon_yuv if (recon_yuv and recon_yuv.is_file()) else None,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            total_bits=metrics.get("total_bits"),
            bitrate_kbps=metrics.get("bitrate_kbps"),
            psnr_y=metrics.get("psnr_y"),
            psnr_u=metrics.get("psnr_u"),
            psnr_v=metrics.get("psnr_v"),
            psnr_yuv=metrics.get("psnr_yuv"),
            encode_time_sec=metrics.get("encode_time_sec"),
            frames_encoded=metrics.get("frames_encoded", frame_count),
        )

    def decode(
        self,
        bitstream_path: Union[str, Path],
        output_recon_path: Union[str, Path],
        output_bit_depth: int = 8,
        extra_args: Optional[List[str]] = None,
    ) -> VTMDecodeResult:
        """
        Run VTM 12.0 Decoder on a .vvc bitstream.

        Args:
            bitstream_path: Path to input .vvc bitstream.
            output_recon_path: Path where reconstructed YUV will be saved.
            output_bit_depth: Reconstruction bit depth (default 8).
            extra_args: Optional additional command-line flags.

        Returns:
            VTMDecodeResult containing execution details.
        """
        bitstream_path = Path(bitstream_path)
        output_recon_path = Path(output_recon_path)

        if not bitstream_path.is_file():
            raise FileNotFoundError(f"Bitstream file does not exist: {bitstream_path}")

        output_recon_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.decoder_path),
            "-b", str(bitstream_path),
            "-o", str(output_recon_path),
            "-d", str(output_bit_depth),
        ]

        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self._env,
        )

        if proc.returncode != 0:
            raise VTMExecutionError(
                f"VTM Decoder execution failed with code {proc.returncode}",
                executable=self.decoder_path,
                cmd=cmd,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                expected_output=output_recon_path,
            )

        if not output_recon_path.is_file() or output_recon_path.stat().st_size == 0:
            raise VTMExecutionError(
                f"VTM Decoder finished with returncode 0 but reconstructed file was not created or is empty: {output_recon_path}",
                executable=self.decoder_path,
                cmd=cmd,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                expected_output=output_recon_path,
            )

        metrics = self.parse_decoder_stdout(proc.stdout)

        return VTMDecodeResult(
            recon_path=output_recon_path,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            frames_decoded=metrics.get("frames_decoded"),
            decode_time_sec=metrics.get("decode_time_sec"),
        )

    @staticmethod
    def parse_encoder_stdout(stdout: str) -> Dict[str, Any]:
        """
        Parse VTM encoder stdout to extract coding metrics.

        Extracts:
            - frames_encoded: Total coded frames count
            - bitrate_kbps: Reported summary bitrate in kbps
            - psnr_y, psnr_u, psnr_v, psnr_yuv: Component PSNRs
            - total_bits: Sum of POC bits
            - encode_time_sec: Total encoding elapsed seconds
        """
        result: Dict[str, Any] = {}

        # Parse per-POC bit counts: e.g. "POC    0 ... 2096 bits"
        poc_bits = re.findall(r"POC\s+\d+.*?\s+(\d+)\s+bits", stdout)
        if poc_bits:
            result["total_bits"] = sum(int(b) for b in poc_bits)
            result["frames_encoded"] = len(poc_bits)

        # Parse summary table:
        # \tTotal Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR
        # \t        4    a      19.2000   52.1898  999.9900  999.9900   53.8914
        summary_match = re.search(
            r"Total Frames\s*\|\s*Bitrate\s+Y-PSNR\s+U-PSNR\s+V-PSNR\s+YUV-PSNR\s*\n\s*(\d+)\s+[a-z]?\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",
            stdout,
        )
        if summary_match:
            result["frames_encoded"] = int(summary_match.group(1))
            result["bitrate_kbps"] = float(summary_match.group(2))
            result["psnr_y"] = float(summary_match.group(3))
            result["psnr_u"] = float(summary_match.group(4))
            result["psnr_v"] = float(summary_match.group(5))
            result["psnr_yuv"] = float(summary_match.group(6))

        # Parse Total Time: "Total Time:        1.798 sec. [user]        1.799 sec. [elapsed]"
        time_match = re.search(r"Total Time:\s+([\d\.]+)\s+sec\..*?([\d\.]+)\s+sec\.\s+\[elapsed\]", stdout)
        if time_match:
            result["encode_time_sec"] = float(time_match.group(2))
        else:
            time_match_simple = re.search(r"Total Time:\s+([\d\.]+)\s+sec\.", stdout)
            if time_match_simple:
                result["encode_time_sec"] = float(time_match_simple.group(1))

        return result

    @staticmethod
    def parse_decoder_stdout(stdout: str) -> Dict[str, Any]:
        """Parse VTM decoder stdout for decoded frames and timing."""
        result: Dict[str, Any] = {}

        pocs = re.findall(r"POC\s+\d+", stdout)
        if pocs:
            result["frames_decoded"] = len(pocs)

        time_match = re.search(r"Total Time:\s+([\d\.]+)\s+sec\.", stdout)
        if time_match:
            result["decode_time_sec"] = float(time_match.group(1))

        return result

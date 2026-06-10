import argparse
import os
from glob import glob
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


TARGET_SR = 16000
HIGH_SR = 48000
DEFAULT_PATTERNS = ["**/*.wav", "**/*.WAV", "**/*.mp3", "**/*.MP3"]


def process_one_file(
    in_path: str,
    in_root: str,
    out_root: str,
    target_sr: int = TARGET_SR,
    high_sr: int = HIGH_SR,
    duration: Optional[float] = None,
    skip_if_exists: bool = True,
    always_updown: bool = True,
) -> bool:
    """
    Convert one audio file to mono float32 WAV at the target sampling rate.

    The default pipeline uses an optional upsample-then-downsample chain so that
    original 16 kHz files and higher-rate files follow a more consistent
    resampling path.
    """
    rel_path = os.path.relpath(in_path, in_root)
    rel_no_ext = os.path.splitext(rel_path)[0]
    out_path = os.path.join(out_root, rel_no_ext + ".wav")

    if skip_if_exists and os.path.exists(out_path):
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    waveform, original_sr = librosa.load(
        in_path,
        sr=None,
        mono=True,
        duration=duration,
        dtype=np.float32,
    )
    waveform = waveform.astype(np.float32)

    if always_updown:
        mid_sr = high_sr if original_sr != high_sr else original_sr
        if original_sr != high_sr:
            waveform = librosa.resample(
                waveform,
                orig_sr=original_sr,
                target_sr=high_sr,
                res_type="soxr_hq",
            )
        if mid_sr != target_sr:
            waveform = librosa.resample(
                waveform,
                orig_sr=mid_sr,
                target_sr=target_sr,
                res_type="soxr_hq",
            )
    else:
        if original_sr <= target_sr:
            mid_sr = high_sr if original_sr != high_sr else original_sr
            if original_sr != high_sr:
                waveform = librosa.resample(
                    waveform,
                    orig_sr=original_sr,
                    target_sr=high_sr,
                    res_type="soxr_hq",
                )
            if mid_sr != target_sr:
                waveform = librosa.resample(
                    waveform,
                    orig_sr=mid_sr,
                    target_sr=target_sr,
                    res_type="soxr_hq",
                )
        elif original_sr != target_sr:
            waveform = librosa.resample(
                waveform,
                orig_sr=original_sr,
                target_sr=target_sr,
                res_type="soxr_hq",
            )

    sf.write(out_path, waveform.astype(np.float32), target_sr, subtype="FLOAT")
    return True


def collect_files(in_root: str, patterns: list[str]) -> list[str]:
    file_list: list[str] = []
    for pattern in patterns:
        file_list.extend(glob(os.path.join(in_root, pattern), recursive=True))
    return sorted(set(file_list))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-convert audio files to mono float32 WAV at 16 kHz."
    )
    parser.add_argument("--in_root", type=str, required=True, help="Input audio root directory")
    parser.add_argument("--out_root", type=str, required=True, help="Output WAV root directory")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=DEFAULT_PATTERNS,
        help="Recursive glob patterns used to collect input files",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional maximum duration in seconds",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=TARGET_SR,
        help="Target sampling rate",
    )
    parser.add_argument(
        "--high_sr",
        type=int,
        default=HIGH_SR,
        help="Intermediate sampling rate for the up-down pipeline",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Overwrite files even if the output already exists",
    )
    parser.add_argument(
        "--direct_resample",
        action="store_true",
        help="Disable the up-then-down resampling chain and resample directly",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    in_root = str(Path(args.in_root))
    out_root = str(Path(args.out_root))
    file_list = collect_files(in_root, args.patterns)

    print(f"Found {len(file_list)} files under {in_root}")

    processed = 0
    skipped = 0
    for path in tqdm(file_list, desc="Preprocessing"):
        try:
            ok = process_one_file(
                in_path=path,
                in_root=in_root,
                out_root=out_root,
                target_sr=args.target_sr,
                high_sr=args.high_sr,
                duration=args.duration,
                skip_if_exists=(not args.no_skip),
                always_updown=(not args.direct_resample),
            )
            if ok:
                processed += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"[ERROR] Failed on {path}: {exc}")

    print(f"Done. processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
